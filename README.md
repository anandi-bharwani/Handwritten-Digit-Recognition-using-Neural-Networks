t# Handwritten-Digit-Recognition-using-Neural-Networks

The Kaggle Digit Reognizer data set, i.e, the MNIST data set has been used for this peoject. It contains images of size 28x28 pixels each.

I've implemented two optimization techniques on the vanilla ANN: momentum and RMSProp. Also, I've done the code in both Theano and TensorFlow for learning purpose. With these I got a maximum score of 0.97200 in Kaggle. [theanoANN.py]

This was improved when I added 2 Convolution-Pooling Layers before the fully connected ANN which got a score of 0.99014 in Kaggle. [theanoCNN.py]
