import numpy as np
from identification_shallownn.matrix_manip import normalize

def _find_rank_one_matrix_step(X, Proj,d, gamma=2):
    X = np.reshape(X, (d,d))
    U,D,V = np.linalg.svd(X)
    D[0]*= gamma
    X = U.dot(np.diag(D).dot(V))
    X = Proj.dot(np.reshape(X, d**2))
    X = normalize(X)
    return X
    

def find_rank_one_matrix(X_0, Proj, n_steps, gamma=2):
    assert Proj.shape[0] == Proj.shape[1] and X_0.shape[0] == X_0.shape[1], "The projection and the input matrix have to be symmetric"
    assert Proj.shape[0] == X_0.shape[0]**2, "Dimensions do not align."
    
    #Make sure the initial value lies on the intersection of the subspace with the unit-sphere
    d = X_0.shape[1]
    #Flatten & making sure that X_0 lies on the subspace
    X_0 = np.reshape(X_0, d**2)
    X = Proj.dot(X_0)
    X = normalize(X)
    for _ in range(n_steps):
        X = _find_rank_one_matrix_step(X, Proj, d, gamma)
    return np.reshape(X,(d,d)) 
