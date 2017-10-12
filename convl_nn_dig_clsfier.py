import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

no_classes = 10
batch_size = 50

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)


def get_weights(shape):
	W = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
	return W

def get_bias(shape):
	b = tf.Variable(tf.constant(0.1, shape = shape))
	return b

def conv2d(x,Weights):
	conv_lay = tf.nn.conv2d(x, Weights, strides=[1, 1, 1, 1], padding = 'SAME')
	return conv_lay

def max_pool_2x2(x):
	pool = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	return pool

x_images = tf.reshape(x, shape = [-1, 28, 28, 1])


def conv_nn_model(x):

	#randomly initializing weights and biases of convolution, fully connected and output layers...
	W_conv_1 = get_weights([5, 5, 1, 32])	
	b_conv_1 = get_bias([32])

	W_conv_2 = get_weights([5, 5, 32, 64])
	b_conv_2 = get_bias([64])

	W_fc = get_weights([7*7*64, 1024])
	b_fc = get_bias([1024])

	W_out = get_weights([1024, no_classes])
	b_out = get_bias([no_classes])


	#first convolution layer + ReLU and Pool layer...
	conv_layer_1 = tf.nn.relu(conv2d(x_images, W_conv_1) + b_conv_1)
	pool_1 = max_pool_2x2(conv_layer_1)

	#second convolution layer + ReLU and Pool layer...
	conv_layer_2 = tf.nn.relu(conv2d(pool_1, W_conv_2) + b_conv_2)
	pool_2 = max_pool_2x2(conv_layer_2)

	#Reshaping the pool_2 to 1D tensor...
	pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])

	#Now, the fully connected layer...
	fc_layer = tf.nn.relu(tf.matmul(pool_2_flat, W_fc) + b_fc)

	#Applying dropout before the output layer...
	keep_prob = tf.placeholder(tf.float32)
	keep_prob = 0.8
	fc_layer_drop = tf.nn.dropout(fc_layer, keep_prob)

	#Finally, the output layer...
	out_layer = tf.matmul(fc_layer_drop, W_out) + b_out

	return out_layer

def train_nn(x):
	prediction = conv_nn_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

	#Optimizing function analogous to gradient descent...
	optmizer = tf.train.AdamOptimizer().minimize(cost)

	no_iter = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for i in range(no_iter):
			cost_dec = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				train_x, train_y = mnist.train.next_batch(batch_size) #function to iterate through batches...
				_, c = sess.run([optmizer, cost], feed_dict= {x: train_x, y: train_y})
				cost_dec += c
			print('Iteration',i+1,'out of',no_iter,'completed & Cost:',cost_dec)

		correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		print('Model\'s Accuracy:',sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

train_nn(x)
