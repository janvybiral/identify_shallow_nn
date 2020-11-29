import numpy as np
from identification_shallownn.SNN import SNN, generate_random_SNN
import identification_shallownn.matrix_manip as mm
from identification_shallownn.matrix_algorithms import find_rank_one_matrix 
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


def approximate_space_svd(hess_apx, m):
    U, _, _ = np.linalg.svd(hess_apx)
    P = U[:,:m]
    P = P.dot(P.T)
    return P,U[:,:m]


def approximate_one_vector(proj,U, n_steps):
    d = int(np.sqrt(U.shape[0]))
    m = U.shape[1]
    X_0 = U.dot(np.random.normal(size=m))
    #X_0 = hess_apx.T.dot(np.random.normal(size=N))
    X_0 = mm.normalize(X_0)
    X_0 = X_0.reshape((d,d))
    result = find_rank_one_matrix(X_0=X_0, Proj=proj, n_steps=n_steps, gamma = 2)
    Ur, _, _ = np.linalg.svd(result)
    return Ur[:, 0]


def a_cand_SNN(net : SNN, X, n_rep, n_steps_apgd = 15):
    hessians = net.approx_ddf(X)
    proj, U = approximate_space_svd(hessians.T, net.m)
    a_cand = []
    for _ in range(n_rep):
        a_cand.append(approximate_one_vector(proj, U, n_steps_apgd))
    return a_cand


def apply_gamma(Z : np.ndarray, d, gamma) -> np.ndarray:
    for c_id in range(Z.shape[1]):
        v,_,_ = randomized_svd(Z[:,c_id].reshape(d,d), n_components=1)
        Z[:,c_id] = Z[:,c_id] + gamma * np.tensordot(v,v, axes=0).reshape(Z.shape[0])
    return Z

def cluster_a_cand(arr, m):
    #Align & cluster with KMeans
    for i, a in enumerate(arr):
        ind = np.argmax(np.abs(a))
        if a[ind] < 0:
            arr[i] = - arr[i]
    data = np.array(arr)
    kmean = KMeans(n_clusters = m).fit(data)
    return kmean.cluster_centers_

def propose_A(net: SNN, Anew):
    Aorig = net.A
    Acorrected = np.zeros(Anew.shape)
    for i in range(Anew.shape[1]):
        ind = np.argmax(np.abs(Aorig.T.dot(Anew[:, i])))
        if Aorig.T.dot(Anew[:, i])[ind] < 0:
            Acorrected[:, ind] = -Anew[:, i]
        else:
            Acorrected[:, ind] = Anew[:, i]
    return Acorrected


def identify_weights(net:SNN, X, n_rep, parallel = False):
    a_cand = a_cand_SNN(net, X, n_rep)
    Anew = cluster_a_cand(a_cand, net.m).T
    return propose_A(net, Anew)



#######Start gradient descent 

def gradient_descent_A(X, Y, net:SNN, epochs, learning_rate, alpha = 0):
    m = net.m
    losses = []
    for _ in range(epochs):
        momentum = np.zeros((m,m))
        loss_p = (net.eval(X) - Y)
        for k in range(m):
            update =  2*X.T * loss_p * net.b[k] * net.dg(X.dot(net.A[:,k]) + net.s1[k])
            update = np.mean(update, axis=1)
            momentum[:, k] = update
        net.A = net.A - learning_rate * momentum
        loss = np.mean(loss_p**2)
        losses.append(loss)
    return losses
