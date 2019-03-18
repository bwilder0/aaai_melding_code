
import torch
class ContinuousOptimizer(torch.autograd.Function):
    """
    pytorch module for differentiable submodular maximization. The forward pass 
    computes the optimal x for given parameters. The backward pass differentiates 
    that optimal x wrt the parameters.
    """
    
    def __init__(self, optimize_func, get_dgradf_dparams, get_hessian=None, max_x = 1.):
        super(ContinuousOptimizer, self).__init__()
        self.optimize_func = optimize_func
        self.get_dgradf_dparams = get_dgradf_dparams
        self.verbose = True
        self.get_hessian = get_hessian
        self.all_xs = []
        self.max_x = max_x
        
    def forward(self, params):
        """
        Computes the optimal x using the supplied optimizer. 
        """
        import numpy as np
        with torch.enable_grad():
            x = self.optimize_func(params, verbose=self.verbose)
        self.x =  x.data
        self.all_xs.append(self.x.detach().numpy())
        self.params = params
        self.xgrad = x.grad.data
        return x.data

    def backward(self, grad_output):
        """
        Differentiates the optimal x returned by the forward pass with respect
        to the ratings matrix that was given as input.
        """
        import numpy as np
        from torch.autograd import Variable
        x = self.x
        params = self.params
        xgrad = self.xgrad
        dxdr = self.get_dxdr(x.detach().numpy(), -xgrad.detach().numpy(), params.detach().numpy(), self.get_dgradf_dparams, self.get_hessian, self.max_x)
        dxdr_t = torch.from_numpy(np.transpose(dxdr))
        out = torch.mm(dxdr_t.float(), grad_output.view(len(x), 1)) 
        return out.view_as(params)  
    
    @staticmethod
    def get_dxdr(x, grad, params, get_dgradf_dparams, get_hessian, max_x):
        '''
        Returns the derivative of the optimal solution in the region around x in 
        terms of the rating matrix r. 
        
        x: an optimal solution
        
        grad: df/dx at x
        
        params: the current parameter settings
        '''
        import numpy as np
        import scipy as sp
        import scipy.sparse
        import scipy.linalg
        n = len(x)
        #first get the optimal dual variables via the KKT conditions
        #dual variable for constraint sum(x) <= k
        if np.logical_and(x > 0, x < max_x).any():
            lambda_sum = np.mean(grad[np.logical_and(x > 0, x < max_x)])
        else:
            lambda_sum = 0
        #dual variable for constraint x <= max_x
        lambda_upper = []
        #dual variable for constraint x >= 0
        lambda_lower = []
        for i in range(n):
            if np.abs(x[i] - max_x) < 0.000001:
                lambda_upper.append(grad[i] - lambda_sum)
            else:
                lambda_upper.append(0)
            if x[i] > 0:
                lambda_lower.append(0)
            else:
                lambda_lower.append(grad[i] - lambda_sum)
        #number of constraints
        m = 2*n + 1
        #collect value of dual variables
        lam = np.zeros((m))
        lam[0] = lambda_sum
        lam[1:(n+1)] = lambda_upper
        lam[n+1:] = lambda_lower
        diag_lambda = np.matrix(np.diag(lam))
        #collect value of constraints
        g = np.zeros((m))
        #TODO: replace the second x.sum() with k so that this is actually generally correct
        g[0] = x.sum() - x.sum()
        g[1:(n+1)] = x - max_x
        g[n+1:] = -x
        diag_g = np.matrix(np.diag(g))
        #gradient of constraints wrt x
        dgdx = np.zeros((m, n))
        #gradient of constraint sum(x) <= k
        dgdx[0, :] = 1
        #gradient of constraints x <= 1
        for i in range(1, n+1):
            dgdx[i, i-1] = 1
        #gradient of constraints x >= 0 <--> -x <= 0
        for i in range(n+1, m):
            dgdx[i, i-(n+1)] = -1
        dgdx = np.matrix(dgdx)
        #the Hessian matrix -- all zeros for now
        if get_hessian == None:
            H = np.matrix(np.zeros((n,n)))
        else:
            H = get_hessian(x, params)
        #coefficient matrix for the linear system
        A = np.bmat([[H, np.transpose(dgdx)], [diag_lambda*dgdx, diag_g]])
        #add 0.01*I to improve conditioning
        A = A + 0.01*np.eye(n+m)
        #RHS of the linear system, mostly partial derivative of grad f wrt params
        dgradf_dparams = get_dgradf_dparams(x, params, num_samples = 1000)
        reshaped = np.zeros((dgradf_dparams.shape[0], dgradf_dparams.shape[1]*dgradf_dparams.shape[2]))
        for i in range(n):
            reshaped[i] = dgradf_dparams[i].flatten()
        b = np.bmat([[reshaped], [np.zeros((m, reshaped.shape[1]))]])
        #solution to the system
        derivatives = sp.linalg.solve(A, b)
        if np.isnan(derivatives).any():
            print('report')
            print(np.isnan(A).any())
            print(np.isnan(b).any())
            print(np.isnan(dgdx).any())
            print(np.isnan(diag_lambda).any())
            print(np.isnan(diag_g).any())
            print(np.isnan(dgradf_dparams).any())
        #first n are derivatives of primal variables
        derivatives = derivatives[:n]
        return derivatives
