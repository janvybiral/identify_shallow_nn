import numpy as np 


def normalize(v):
    return v / np.linalg.norm(v)

def normalize_col(A):
    for i in range(A.shape[1]):
        A[:, i] = A[:,i] / np.linalg.norm(A[:,i])
    return A

def disturbation(A,disturb):
    _,m=A.shape
    U, D, V = np.linalg.svd(A)
    #print("U: {}, D: {}, V: {}".format(
    #    U.shape, D.shape, V.shape
    #))
    Atemp = U[:, :m].dot(np.diag(D + disturb)).dot(V)
    for i in range(m):
        Atemp[:, i] = normalize(Atemp[:,i])
    _, D, _ = np.linalg.svd(Atemp)
    
    return np.linalg.norm(D-np.ones(m))

def creating_nonQO(A, eps_A, seed=None):
    np.random.seed(seed)
    U,D,V = np.linalg.svd(A)
    m = A.shape[1]
    tol = 0.001
    I = np.ones(m)
    pert = np.random.normal(size = m)
    l = np.linalg.norm(pert)
    l_low = 0
    max_iterations = 100
    k = 0
    while True:
        xi = (l - l_low)/2 + l_low
        res = disturbation(A, pert*xi) - eps_A
        if np.abs(res) < tol or k > max_iterations:
            break
        elif res > 0:
            l = xi
        else:
            l_low = xi
        k = k+1
    Anew = U[:,:m].dot(np.diag(D+pert*xi)).dot(V)
    for i in range(m):
        Anew[:,i] = normalize(Anew[:,i])
    return Anew
