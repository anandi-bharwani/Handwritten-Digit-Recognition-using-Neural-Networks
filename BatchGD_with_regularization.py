import numpy as np
from util import get_normalized_data, y2indicator, forward_prop, derivative_b1, derivative_w1, derivative_b2, derivative_w2, error_rate, cost, forward_relu, get_test_data

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
    reg = 0.01

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
            W2 += learning_rate * (derivative_w2(pY, YBatch_ind, Z) + reg*W2)
            b2 += learning_rate * (derivative_b2(pY, YBatch_ind) + reg*b2)
            W1 += learning_rate * (derivative_w1(pY, YBatch_ind, W2, Z, XBatch) + reg*W1)
            b1 += learning_rate * (derivative_b1(pY, YBatch_ind, W2, Z) + reg*b1)

            if n%100 == 0:
                #Forward prop
                #pY, Z = forward_relu(XBatch, W1, b1, W2, b2)
                YBatch = Y[n*batchSz:n*batchSz + batchSz]
                P = np.argmax(pY, axis=1)
                er = error_rate(P, YBatch)

                c = cost(YBatch_ind, pY)
                print("Loop: ", i, n, "Error rate: ", er, "Cost: ", c )
        
    
    pY, Z = forward_relu(X, W1, b1, W2, b2)
    p = np.argmax(pY, axis=1)
    print("Final training error rate: ", error_rate(p, Y))

    XTest = get_test_data()
    pY, ZTest = forward_relu(XTest, W1, b1, W2, b2)
    YTest = np.argmax(pY, axis=1)

    f = open("test_result.csv","w")
    f.write("ImageId,Label\n")
    n = YTest.shape[0]
    for i in range(n):
        f.write(str(i+1) + "," + str(YTest[i]) + "\n")
    f.close()

def main():
    batch_grad()
    #X = get_test_data()

if __name__ == "__main__":
    main()



