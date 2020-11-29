"""
SNN.py defines shallow neural networks 
"""

import numpy as np
from identification_shallownn.matrix_manip import creating_nonQO
from scipy.stats import ortho_group


class SNN: 
    def __init__(self, A, b, s1, s2, act, dact):
        self.A = A
        self.b = b
        self.s1 = s1
        self.s2 = s2 
        self.g = act
        self.dg = dact
        self.d, self.m  = self.A.shape

        
    def eval(self, X):
        return self.g(X.dot(self.A) + self.s1).dot(self.b) + self.s2

    def approx_df(self, X):
        pass
        
    def approx_ddf(self, X, eps=0.001):
        ddf = np.zeros(shape=(self.d,self.d))
        ddfs = np.zeros(shape=(X.shape[0], self.d * self.m))
        e = np.diag(np.ones(self.m))
        for k,x in enumerate(X):
            for j in range(self.m):
                for i in range(j+1):
                    upd = self.eval(x + eps*e[i] + eps*e[j]) - self.eval(x + eps*e[i]) - self.eval(x+eps*e[j]) + self.eval(x)
                    ddf[i,j] = upd/(eps**2)
                    ddf[j,i] = upd/(eps**2)
            ddfs[k] = (ddf/2).reshape(self.d * self.m)
        return ddfs

def generate_random_SNN(d,m, eps_A=0,act=np.tanh, dact=lambda x : 1 / np.cosh(x)**2, seed=None) -> SNN:
    np.random.seed(seed)
    s2 = 0
    s1 = np.random.normal(0,0.2, size = m, )
    b = np.random.normal(1,0.2,size = m)
    b = b / np.linalg.norm(b)
    A = ortho_group.rvs(d, random_state=seed)[:,:m]
    if eps_A > 0.0:
        A = creating_nonQO(A, eps_A, seed=seed)
    net = SNN(A = A, b=b, s1 = s1, s2 = s2, act = np.tanh, dact = dact)
    return net