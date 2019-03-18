def project_uniform_matroid_boundary(x, k, c=1):
    '''
    Exact projection algorithm of Karimi et al. This is the projection implementation
    that should be used now.
    
    Projects x onto the set {y: 0 <= y <= 1/c, ||y||_1 = k}
    '''
    import numpy as np
    k *= c
    n = len(x)
    x = x.copy()
    alpha_upper = x/c
    alpha_lower = (x*c - 1)/c**2
    S = []
    S.extend(alpha_lower)
    S.extend(alpha_upper)
    S.sort()
    S = np.unique(S)
    h = n
    alpha = min(S) - 1
    m = 0
    for i in range(len(S)):
        hprime = h + (S[i] - alpha)*m
        if hprime < k and k <= h:
            alphastar = (S[i] - alpha)*(h - k)/(h - hprime) + alpha
            result = np.zeros((n))
            for j in range(n):
                if alpha_lower[j] > alphastar:
                    result[j] = 1./c
                elif alpha_upper[j] >= alphastar:
                    result[j] = x[j] - alphastar*c
            return result
        m -= (alpha_lower == S[i]).sum()*(c**2)
        m += (alpha_upper == S[i]).sum()*(c**2)
        h = hprime
        alpha = S[i]
    raise Exception('projection did not terminate')

def project_cvx(x, k):
    '''
    Exact Euclidean projection onto the boundary of the k uniform matroid polytope.
    '''
    from cvxpy import Variable, Minimize, sum_squares, Problem
    import numpy as np
    n = len(x)
    p = Variable(n, 1)
    objective = Minimize(sum_squares(p - x))
    constraints = [sum(p) == k, p >= 0, p <= 1]
    prob = Problem(objective, constraints)
    prob.solve()
    return np.reshape(np.array(p.value), x.shape)
