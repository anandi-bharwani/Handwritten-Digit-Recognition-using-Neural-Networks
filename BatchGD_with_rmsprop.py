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

    learning_rate = 0.001
    reg = 0.01
    cache_w2 = 0
    cache_b2 = 0
    cache_w1 = 0
    cache_b1 = 0
    decay_rate = 0.999
    eps = 10e-10

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
            gW2 = derivative_w2(pY, YBatch_ind, Z) + reg*W2
            cache_w2 = decay_rate*cache_w2 + (1-decay_rate)*gW2*gW2
            W2 += learning_rate * gW2 / (np.sqrt(cache_w2) + eps)

            gb2 = derivative_b2(pY, YBatch_ind) + reg*b2
            cache_b2 = decay_rate*cache_b2 + (1-decay_rate)*gb2*gb2
            b2 += learning_rate * gb2 / (np.sqrt(cache_b2) + eps)

            gW1 = derivative_w1(pY, YBatch_ind, W2, Z, XBatch) + reg*W1
            cache_b2 = decay_rate*cache_b2 + (1-decay_rate)*gb2*gb2
            b2 += learning_rate * gb2 / (np.sqrt(cache_b2) + eps)
            
            gb1 = derivative_b1(pY, YBatch_ind, W2, Z) + reg*b1
            cache_b1 = decay_rate*cache_b1 + (1-decay_rate)*gb1*gb1
            b1 += learning_rate * gb1 / (np.sqrt(cache_b1) + eps)
            

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
    print("Final Final training error rate: ", error_rate(p, Y))

    XTest = get_test_data()
    pY, ZTest = forward_relu(XTest, W1, b1, W2, b2)
    YTest = np.argmax(pY, axis=1)

    f = open("test_rms.csv","w")
    f.write("ImageId,Label\n")
    n = YTest.shape[0]
    for i in range(n):
        f.write(str(i+1) + "," + str(YTest[i]) + "\n")
    f.close()

def main():
    batch_grad()

if __name__ == "__main__":
    main()



