import theano.tensor as T
import theano
import numpy as np
from util import get_normalized_data, y2indicator

def relu(A):
	return A*(A>0)

def error_rate(p, y):
	return np.mean(p!=y)

def main():
	#Get the data and define usual variables
	X, Y = get_normalized_data()
	XTrain = X[:-1000, :]
	YTrain = Y[:-1000]
	YTrain_ind = y2indicator(YTrain)
	XTest = X[-1000:, :]
	YTest = Y[-1000:]
	YTest_ind = y2indicator(YTest)
	Y_ind = y2indicator(Y)

	#Initialize random weights
	lr = 0.00001
	N, D = XTrain.shape
	K = len(set(Y))
	M = 300
	
	W1_val = np.random.randn(D, M)/np.sqrt(D+M)
	b1_val = np.zeros(M)
	W2_val = np.random.randn(M, K)/np.sqrt(M+K)
	b2_val = np.zeros(K)

	#Define Theano variables and functions	
	thX = T.matrix('X')
	thY = T.matrix('Y')

	W1 = theano.shared(W1_val, 'W1')
	b1 = theano.shared(b1_val, 'b1')
	W2 = theano.shared(W2_val, 'W2')
	b2 = theano.shared(b2_val, 'b2')

	Z = relu(thX.dot(W1) + b1)
	pY = T.nnet.softmax(Z.dot(W2) + b2)

	cost = -(thY * T.log(pY)).sum()
	pred = T.argmax(pY, axis=1)

	W1_update = W1 - lr*T.grad(cost, W1)
	b1_update = b1 - lr*T.grad(cost,b1)
	W2_update = W2 - lr*T.grad(cost, W2)
	b2_update = b2 - lr*T.grad(cost,b2)

	train = theano.function(
		inputs=[thX,thY], 
		updates=[(W1, W1_update), (b1, b1_update), (W2, W2_update), (b2, b2_update)],
	)

	get_cost = theano.function(
		inputs=[thX,thY],
		outputs=[cost, pred],
	)

	
	#Run Batch GD using the theano functions
	batchSz = 500
	no_batches = int(N/batchSz)
	LL = []
    
	for i in range(5):
		for n in range(no_batches):
    		#get current batch
			XBatch=XTrain[n*batchSz:(n*batchSz + batchSz), :]
			YBatch_ind = YTrain_ind[n*batchSz:(n*batchSz + batchSz), :]
			
			train(XBatch, YBatch_ind)		#Updates weights

			if n%100 == 0:
				#YBatch = Y[n*batchSz:n*batchSz + batchSz]
				cost_val, pred = get_cost(XTest, YTest_ind)
				err = error_rate(pred, YTest)
				print("Cost: ", cost_val, " Error rate: ", err)
				LL.append(cost_val)

	plt.plot(LL)
	plt.show()

if __name__=='__main__':
	main()


