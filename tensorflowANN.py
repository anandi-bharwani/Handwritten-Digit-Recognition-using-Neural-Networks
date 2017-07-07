import tensorflow as tf 
import numpy as np 

A = tf.placeholder(tf.float32)
v = tf.placeholder(tf.float32)

w = matmul(A,v)

with tf.Session() as session:
	output = session.run(w, feeddict={A: np.random.randn(5,5), v: np.random.randn(5,1)})
	print(output, type(output))
