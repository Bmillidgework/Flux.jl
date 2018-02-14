import tensorflow as tf
import numpy as np

# ------------------------------------------------------- #
#            Logistic Regression from scratch             #
# ------------------------------------------------------- #

sess = tf.Session()

x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

init = tf.global_variables_initializer()
sess.run(init)

# See an example prediction
sess.run(pred, feed_dict = {x: np.random.rand(1,784)})

sess.close()

# ------------------------------------------------------- #
#                  Custom Layer: Dense                    #
# ------------------------------------------------------- #

def dense_layer(x, input, out, act = tf.sigmoid):
    W = tf.get_variable("weights", [input, out],
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", [1, out],
                        initializer=tf.random_normal_initializer())
    return act(tf.matmul(x, W) + b)

sess = tf.Session()

x = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope("layer1"):
    y = dense_layer(x, 10, 5)

init = tf.global_variables_initializer()
sess.run(init)
sess.run(y, feed_dict = {x: np.random.rand(1,10)})

sess.close()

# ------------------------------------------------------- #
#                  RNN from scratch                       #
# ------------------------------------------------------- #

input = 10
hidden = 5
length = 13

def step(hprev, x):
    # params
    Wi = tf.get_variable('W', shape=[input, hidden], initializer=tf.random_normal_initializer())
    Wh = tf.get_variable('U', shape=[hidden, hidden], initializer=tf.random_normal_initializer())
    b = tf.get_variable('b', shape=[hidden], initializer=tf.constant_initializer(0.))
    # current hidden state
    h = tf.tanh(tf.matmul(hprev, Wh) + tf.matmul(x,Wi) + b)
    return h

sess = tf.Session()

# (seqlength, batch, features)
xs = tf.placeholder(tf.float32, [length, 1, input])
h = tf.placeholder(tf.float32, [None, hidden])

states = tf.scan(step, xs, initializer=h)

init = tf.global_variables_initializer()
sess.run(init)
sess.run(states, feed_dict = {xs: np.random.rand(length,1,input), h: np.random.randn(1,hidden)})

sess.close()

# ------------------------------------------------------- #
#                  Recursive Net                          #
# ------------------------------------------------------- #

# Too long to repeat here, see https://github.com/erickrf/treernn
