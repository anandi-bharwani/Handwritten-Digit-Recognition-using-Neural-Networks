import numpy as np
from util import get_normalized_data, y2indicator, forward_prop, derivative_b1, derivative_w1, derivative_b2, derivative_w2, error_rate, cost, forward_relu

def batch_grad():

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

    no_batches = int(N/batchSz)
    print("No of bathces: ", no_batches)
    for i in range(300):
        for n in range(no_batches):
            #get current batch
            XBatch = X[n*batchSz:(n*batchSz + batchSz), :]
            #YBatch = Y[n*batchSz:n*batchSz + batchSz]
            YBatch_ind = Y_ind[n*batchSz:(n*batchSz + batchSz), :]
            #Forward prop
            pY, Z = forward_relu(XBatch, W1, b1, W2, b2)

            #Backprop
            W2 += learning_rate * derivative_w2(pY, YBatch_ind, Z)
            b2 += learning_rate * derivative_b2(pY, YBatch_ind)
            W1 += learning_rate * derivative_w1(pY, YBatch_ind, W2, Z, XBatch)
            b1 += learning_rate * derivative_b1(pY, YBatch_ind, W2, Z)

            if n%100 == 0:
                #Forward prop
                #pY, Z = forward_relu(XBatch, W1, b1, W2, b2)
                YBatch = Y[n*batchSz:n*batchSz + batchSz]
                P = np.argmax(pY, axis=1)
                er = error_rate(P, YBatch)

                c = cost(YBatch_ind, pY)
                print("Loop: ", i, n, "Error rate: ", er, "Cost: ", c )
        
    # pY, Z = forward_prop(XTrain, W1, b1, W2, b2)
    # P = np.argmax(pY, axis=1)
    # print("Final training error rate: ", error_rate(P, YTrain))
    #
    # pY, Z = forward_prop(XTest, W1, b1, W2, b2)
    # P = np.argmax(pY, axis=1)
    # print("Final testing error rate: ", error_rate(P, YTest))

    pY, Z = forward_relu(X, W1, b1, W2, b2)
    p = np.argmax(pY, axis=1)
    print("Final Final training error rate: ", error_rate(p, Y))

    #X,Y = get_normalized_data("mnist_test.csv")
    #pY, Z = forward_relu(X, W1, b1, W2, b2)
    #p = np.argmax(pY, axis=1)
    #print("Final Final training error rate: ", error_rate(p, Y))

def main():
    batch_grad()

if __name__ == "__main__":
    main()



