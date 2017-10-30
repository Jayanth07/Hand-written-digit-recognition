import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hl1_nodes = 750
hl2_nodes = 750
no_classes = 10
#learn_rate = 0.001
batch_size = 100

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

# Neural network model...
def nn_model(data):
        
        #Initialising thetas for all the layers...
        theta_hid_layer1 = {'weights': tf.Variable(tf.random_normal([784, hl1_nodes])),
                                                'bias': tf.Variable(tf.random_normal([hl1_nodes]))}
                        
        theta_hid_layer2 = {'weights': tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes])),
                                                'bias': tf.Variable(tf.random_normal([hl2_nodes]))}

        theta_out_layer = {'weights': tf.Variable(tf.random_normal([hl2_nodes, no_classes])),
                                                'bias': tf.Variable(tf.random_normal([no_classes]))}

        #Computing the activation cells i.e. a(x) & passing them into sigmoid function (tf.nn.relu)...
        ac1 = tf.add(tf.matmul(data, theta_hid_layer1['weights']), theta_hid_layer1['bias'])
        ac1 = tf.nn.relu(ac1)

        ac2 = tf.add(tf.matmul(ac1, theta_hid_layer2['weights']), theta_hid_layer2['bias'])
        ac2 = tf.nn.relu(ac2)

        #final output i.e. h(x)...
        output = tf.add(tf.matmul(ac2,theta_out_layer['weights']), theta_out_layer['bias'])

        return output

def train_nn(x):
        prediction = nn_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

        #Optimizing function analogous to gradient descent...
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        no_iter = 10

        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                for i in range(no_iter):
                    cost_dec = 0
                    for _ in range(int(mnist.train.num_examples/batch_size)):
                        train_x, train_y = mnist.train.next_batch(batch_size)
                        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
                        cost_dec += c
                    print('Iteration', (i+1),'out of', no_iter,'completed & Cost:', cost_dec)
                #saving the variables...
                saver = tf.train.Saver()
                save_path = saver.save(sess, "/tmp/nn_model.ckpt")
                print('Model saved to: %s'% save_path)
                
                correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                print('Model\'s Accuracy:',sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

train_nn(x)

