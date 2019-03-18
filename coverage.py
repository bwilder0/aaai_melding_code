import torch
import numpy as np
from numba import jit

@jit
def gradient_coverage(x, P, w):
    n = len(w)
    m = len(x)
    grad = np.zeros(m, dtype=np.float32)
    for i in range(n):
        p_fail = 1 - x*P[:,i]
        p_all_fail = np.prod(p_fail)
        for j in range(m):
            grad[j] += w[i] * P[j, i] * p_all_fail/p_fail[j]
    return grad


@jit
def hessian_coverage(x, P, w):
    n = len(w)
    m = len(x)
    hessian = np.zeros((m,m), dtype=np.float32)
    for i in range(n):
        p_fail = 1 - x*P[:,i]
        p_all_fail = np.prod(p_fail)
        for j in range(m):
            for k in range(m):
                hessian[j, k] = -w[i] * P[j, i] * p_all_fail/(p_fail[j] * p_fail[k])
    return hessian


@jit
def objective_coverage(x, P, w):
    n = len(w)
    total = 0
    for i in range(n):
        p_fail = 1 - x*P[:,i]
        p_all_fail = np.prod(p_fail)
        total += w[i] * (1 - p_all_fail)
    return total

class CoverageInstanceMultilinear(torch.autograd.Function):
    """
    Represents a coverage instance with given coverage probabilities
    P and weights w. Forward pass computes the objective value (if evaluate_forward
    is true). Backward computes the gradients wrt decision variables x.
    """
    def __init__(self, P, w, evaluate_forward):
        super(CoverageInstanceMultilinear, self).__init__()
        self.evaluate_forward = evaluate_forward
        if type(P) != np.ndarray:
            P = P.detach().numpy()
        if type(w) != np.ndarray:
            w = w.detach().numpy()
        self.P = P
        self.w = w
        
    def forward(self, x):
        self.x = x.detach().numpy()
        if self.evaluate_forward:
            out = objective_coverage(self.x, self.P, self.w)
        else:
            out = -1
        return torch.tensor(out).float()
                    
    def backward(self, grad_in):
        grad = gradient_coverage(self.x, self.P, self.w)
        return torch.from_numpy(grad).float()*grad_in.float()


def optimize_coverage_multilinear(P, w, verbose=True, k=10, c=1., minibatch_size = None):
    '''
    Run some variant of SGD for the coverage problem with given 
    coverage probabilities P and weights w
    
    '''
    import torch
    from utils import project_uniform_matroid_boundary as project
    
    #objective which will provide gradient evaluations
    coverage = CoverageInstanceMultilinear(P, w, verbose)
    #decision variables
    x = torch.zeros(P.shape[0], requires_grad = True)
    #set up the optimizer
    learning_rate = 0.1
    optimizer = torch.optim.SGD([x], momentum = 0.9, lr = learning_rate, nesterov=True)
    #take projected stochastic gradient steps
    for t in range(10):
        loss = -coverage(x)
        if verbose:
            print(t, -loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x.data = torch.from_numpy(project(x.data.numpy(), k, 1/c)).float()
    return x

@jit
def dgrad_coverage(x, P, num_samples, w):
    n = len(w)
    m = len(x)
    dgrad = np.zeros((m, m, n), dtype=np.float32)
    for i in range(n):
        p_fail = 1 - x*P[:,i]
        p_all_fail = np.prod(p_fail)
        for j in range(m):
            for k in range(m):
                if j == k:
                    dgrad[j, k, i] =  w[i] * p_all_fail/p_fail[j]
                else:
                    dgrad[j, k, i] = -w[i] * x[k] * P[j, i] * p_all_fail/(p_fail[j] * p_fail[k])
    return dgrad

@jit
def dgrad_coverage_stochastic(x, P, num_samples, w, num_real_samples):
    n = len(w)
    m = len(x)
    rand_rows = np.random.choice(list(range(m)), num_real_samples)
    rand_cols = np.random.choice(list(range(n)), num_real_samples)
    
    dgrad = np.zeros((m, m, n), dtype=np.float32)
    
    p_fail = np.zeros((n, m), dtype=np.float32)
    p_all_fail = np.zeros((n), dtype=np.float32)
    for i in range(n):
        p_fail[i] = 1 - x*P[:,i]
        p_all_fail[i] = np.prod(p_fail[i])
    
    for sample in range(num_real_samples):
        k = rand_rows[sample]
        i = rand_cols[sample]
        for j in range(m):
            if j == k:
                dgrad[j, k, i] =  w[i] * p_all_fail[i]/p_fail[i,j]
            else:
                dgrad[j, k, i] = -w[i] * x[k] * P[j, i] * p_all_fail[i]/(p_fail[i,j] * p_fail[i,k])
    return dgrad