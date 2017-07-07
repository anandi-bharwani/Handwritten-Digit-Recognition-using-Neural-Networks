import numpy as np
import pandas as pd

def get_normalized_data():
    df = pd.read_csv("train.csv")
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]

    # mean = X.mean(axis=0, keepdims=True)
    # std = X.std(axis=0, keepdims=True)
    # np.place(std, std == 0, 1)
    # X = (X - mean)/std
    X = X/X.max()
    return X,Y

def get_test_data():
    df = pd.read_csv("test.csv")
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data
    X = X/X.max()
    return X


def cost(T, P):
    np.place(P, P==0, 10e-280)
    return (T * np.log(P)).sum()

def error_rate(P,Y):
    return np.mean(P != Y)

def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    Y_ind = np.zeros([N, K])
    Y = Y.astype(np.int32)
    for i in range(N):
        Y_ind[i, Y[i]] = 1
    return Y_ind

def forward_prop(X, W1, b1, W2, b2):
    Z = 1/(1 + np.exp(- X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    ret = expA / expA.sum(axis=1, keepdims=True)
    return ret, Z

def forward_relu(X, W1, b1, W2, b2):
    Z = X.dot(W1) + b1
    #Z[Z < 0] = 0
    Z = Z * (Z>0)
    #r, c = Z.shape
    # for i in range(r):
    #     for j in range(c):
    #         if(Z[i][j] < 0):
    #             Z[i][j] = 0
    #Z = [0 if i < 0 else i for i in j for j in a]
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    ret = expA / expA.sum(axis=1, keepdims=True)
    return ret, Z

def derivative_w2(Y, T, Z):
    ret = Z.T.dot((T-Y))
    return ret

def derivative_b2(Y, T):
    ret = (T-Y).sum(axis = 0)
    return ret

def derivative_w1(Y, T, W2, Z, X):
    #ret = X.T.dot((T - Y).dot(W2.T) * Z * (1 - Z))     #For sigmoid
    ret = X.T.dot((T - Y).dot(W2.T) * np.sign(Z))            #For relu
    return ret;

def derivative_b1(Y, T, W2, Z):
    #ret = ((T-Y).dot(W2.T)*Z*(1-Z)).sum(axis=0)                #sigmoid
    ret = ((T-Y).dot(W2.T) * np.sign(Z)).sum(axis=0)                 #relu
    return ret
