import numpy as np
import matplotlib.pyplot as plt

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return np.dot(A,B) - C

def problem_1c (A, B, C):
    return A*B + C.T

def problem_1d (x, y):
    return np.dot(x.T,y)
    # we can also use
    # np.inner(np.transpose(x),np.transpose(y))

def problem_1e (A, x):
    return np.linalg.solve(A, x)

def problem_1f (A, x):
    return (np.linalg.solve(A.T,x.T)).T

def problem_1g (A, i):
    condition = np.empty((np.shape(A)[1],),bool)
    condition[::2] = True
    condition[1::2] = False
    return np.sum(A[i], where=condition)
#     We can also compute this by the following but does more unnecessary computation
#     np.sum(A, axis=1, where=np.resize([True, False], np.shape(A)[1]))[i]

def problem_1h (A, c, d):
    k = np.nonzero((A>=c) & (A<=d))
    S = A[k[0], k[1]]
    return np.mean(S)

def problem_1i (A, k):
    W, V = np.linalg.eig(A)
    return V[:, np.argsort(-W)[:k]]


def problem_1j (x, k, m, s):
    return np.random.multivariate_normal(x + m*np.ones(np.size(x)), s*np.identity(np.size(x)),k).T

def problem_1k (A):
    return np.random.permutation(A)

def problem_1l (x):
    return (x - np.mean(x))/np.std(x)

def problem_1m (x, k):
    return np.repeat(x,k,axis=1)

def problem_1n (X):
    mat3D_1 = np.repeat(X[:, :, np.newaxis], np.shape(X)[1], axis=2)
    mat3D_2 = np.swapaxes(mat3D_1,2,1)
    diff = mat3D_2 - mat3D_1
    return np.linalg.norm(diff,axis=0)

def linear_regression (X_tr, y_tr):
    # w = ((XX')^-1)*(Xy)
    X = X_tr.T
    return np.linalg.solve(np.dot(X,X.T),np.dot(X,y_tr))

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))      # X tranpose
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")
    N = np.shape(ytr)[0]
    NN = np.shape(yte)[0]#number of examples

    # analytical soln for linear regression
    w = linear_regression(X_tr, ytr)

    # For training Dataset
    y_hat_tr = np.dot(X_tr,w)
    y_tr_diff = y_hat_tr - ytr
    loss_tr = np.dot(y_tr_diff.T,y_tr_diff)/(2*N)

    # For testing Dataset
    y_hat_te = np.dot(X_te, w)
    y_te_diff = y_hat_te - yte
    loss_te = np.dot(y_te_diff.T, y_te_diff) /(2*NN)

    print("Training Loss:", loss_tr)
    print("Testing Loss:", loss_te)

    # Report fMSE cost on the training and testing data (separately)
    # ...
