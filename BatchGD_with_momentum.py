import numpy as np
from util import get_normalized_data, y2indicator, forward_prop, derivative_b1, derivative_w1, derivative_b2, derivative_w2, error_rate, cost, forward_relu

def bgd_momentum():

    #get data and for test and train sets
    X,Y = get_normalized_data()
    #XTrain = X[:-1000, :]
    #YTrain = Y[:-1000]
    #YTrain_ind = y2indicator(YTrain)
    #XTest = X[-1000:, :]
    #YTest = Y[-1000:]
    # = y2indicator(YTest)
    Y_ind = y2indicator(Y)

    batchSz = 500
    #Initialize random weights
    N, D = X.shape
    K = len(set(Y))
    M = 300
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 10e-5
    reg = 0.01
    mu = 0.9

    dW1 = 0
    db1 = 0
    dW2 = 0
    db2 = 0

    no_batches = int(N/batchSz)
    print("No of bathces: ", no_batches)
    for i in range(300):
        for n in range(no_batches):
            #get current batch
            XBatch = X[n*batchSz:(n*batchSz + batchSz), :]
            YBatch_ind = Y_ind[n*batchSz:(n*batchSz + batchSz), :]
            #Forward prop
            pY, Z = forward_relu(XBatch, W1, b1, W2, b2)

            #Backprop
            dW2 = mu*dW2 + learning_rate * (derivative_w2(pY, YBatch_ind, Z) + reg*W2)
            W2 +=dW2
            db2 = mu*db2 + learning_rate * (derivative_b2(pY, YBatch_ind) + reg*b2)
            b2 += db2
            dW1 = mu*dW1 + learning_rate * (derivative_w1(pY, YBatch_ind, W2, Z, XBatch) + reg*W1)
            W1 +=dW1
            db1 = mu*db1 * learning_rate * (derivative_b1(pY, YBatch_ind, W2, Z) + reg*b1)
            b1 += db1

            if n%100 == 0:
                #Forward prop
                YBatch = Y[n*batchSz:n*batchSz + batchSz]
                P = np.argmax(pY, axis=1)
                er = error_rate(P, YBatch)

                c = cost(YBatch_ind, pY)
                print("Loop: ", i, n, "Error rate: ", er, "Cost: ", c )
        

    pY, Z = forward_relu(X, W1, b1, W2, b2)
    p = np.argmax(pY, axis=1)
    print("Final Final training error rate: ", error_rate(p, Y))

def bgd_nestorov():

    #get data and for test and train sets
    X,Y = get_normalized_data()
    #XTrain = X[:-1000, :]
    #YTrain = Y[:-1000]
    #YTrain_ind = y2indicator(YTrain)
    #XTest = X[-1000:, :]
    #YTest = Y[-1000:]
    # = y2indicator(YTest)
    Y_ind = y2indicator(Y)

    batchSz = 500
    #Initialize random weights
    N, D = X.shape
    K = len(set(Y))
    M = 300
    W1 = np.random.randn(D, M)/np.sqrt(D+M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)/np.sqrt(M+K)
    b2 = np.random.randn(K)

    learning_rate = 10e-5
    reg = 0.01
    mu = 0.9

    dW1 = 0
    db1 = 0
    dW2 = 0
    db2 = 0

    no_batches = int(N/batchSz)
    print("No of bathces: ", no_batches)
    for i in range(300):
        for n in range(no_batches):
            #get current batch
            XBatch = X[n*batchSz:(n*batchSz + batchSz), :]
            YBatch_ind = Y_ind[n*batchSz:(n*batchSz + batchSz), :]
            #Forward prop
            pY, Z = forward_relu(XBatch, W1, b1, W2, b2)

            #W1(t) = W1(t-1) - mu*dW(t)
            w2_tmp = W2 - learning_rate*mu*dW2
            b2_tmp = b2 - learning_rate*mu*db2
            w1_tmp = W1 - learning_rate*mu*dW1
            b1_tmp = b1 - learning_rate*mu*db1

            #Backprop
            dW2 = mu*dW2 + (derivative_w2(pY, YBatch_ind, Z) + reg*w2_tmp)
            W2 += learning_rate * dW2
            db2 = mu*db2 + (derivative_b2(pY, YBatch_ind) + reg*b2_tmp)
            b2 += learning_rate * db2
            dW1 = mu*dW1 + (derivative_w1(pY, YBatch_ind, W2, Z, XBatch) + reg*w1_tmp)
            W1 += learning_rate * dW1
            db1 = mu*db1 * (derivative_b1(pY, YBatch_ind, W2, Z) + reg*b1_tmp)
            b1 += learning_rate * db1

            if n%100 == 0:
                #Forward prop
                YBatch = Y[n*batchSz:n*batchSz + batchSz]
                P = np.argmax(pY, axis=1)
                er = error_rate(P, YBatch)

                c = cost(YBatch_ind, pY)
                print("Loop: ", i, n, "Error rate: ", er, "Cost: ", c )
        

    pY, Z = forward_relu(X, W1, b1, W2, b2)
    p = np.argmax(pY, axis=1)
    print("Final Final training error rate: ", error_rate(p, Y))


def main():
    #bgd_momentum()
    bgd_nestorov()

if __name__ == "__main__":
    main()



