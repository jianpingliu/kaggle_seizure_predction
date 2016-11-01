import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

class Model(object):
    def __init__(self, n_channels, dim1, dim2, classes, l2_reg_lambda=0.0):
        # placeholders
        self.x = tf.placeholder(tf.float32, [None, n_channels, dim1, dim2], name="x")
        self.y = tf.placeholder(tf.float32, [None, classes], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # l2 regularization loss
        l2_loss = tf.constant(0.0)

        # reshape x
        x = tf.transpose(self.x, perm=[0, 2, 3, 1])
        # [batch, dim1, dim2, n_channels]

        # conv layer 1
        self.h1 = self.conv(x, 3, 5, 64, strides=[1, 4])
        self.h1_pool = self.pool(self.h1, 2, strides=2)

        # conv layer2
        self.h2 = self.conv(self.h1_pool, 3, 3, 128, strides=[2, 2])
        self.h2_pool = self.pool(self.h2, 2, strides=2)

        # conv layer3
        self.h3 = self.conv(self.h2_pool, 3, 3, 256, strides=[1, 2])

        # conv layer4
        self.h4 = self.conv(self.h3, 2, 2, 512, strides=[1, 1])

        # BiRNN layer
        self.h5 = self.BiRNN(self.h4, 128, depth=3)
        n_steps = len(self.h5)

        # average n_steps
        self.h6 = tf.add_n(self.h5)/n_steps

        # output
        with tf.name_scope("output"):
            hidden_n = int(list(self.h6.get_shape())[-1])
            W = tf.get_variable(
                "W",
                shape=[hidden_n, classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.01, shape=[classes]), name="b")
            self.scores = tf.sigmoid(tf.nn.xw_plus_b(self.h6, W, b), name="scores")
            self.probability = tf.nn.softmax(self.scores, name="probability")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y)
            self.loss = tf.reduce_mean(losses)

        # accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def conv(self, x, filter_height, filter_width, n_filters, strides=[1,1]):
        in_channels = int(list(x.get_shape())[-1])

        W = tf.Variable(
            tf.truncated_normal(
                [filter_height, filter_width, in_channels, n_filters], stddev=0.01))

        b = tf.Variable(tf.zeros([n_filters]))

        conv = tf.nn.conv2d(x, W, strides=[1, strides[0], strides[1], 1], padding="VALID")

        h = tf.nn.relu(tf.nn.bias_add(conv, b))

        return h

    def pool(self, x, window, strides=2):
        h_pool = tf.nn.max_pool(x, ksize=[1, window, window, 1], 
            strides=[1, strides, strides, 1], padding='VALID')

        return h_pool

    def BiRNN(self, x, dim, depth=1):
        x_shape = list(x.get_shape())
        dim1 = int(x_shape[1])
        dim2 = int(x_shape[3])
        n_steps = int(x_shape[2])
        n_input = dim1*dim2

        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)

        cell = rnn_cell.BasicLSTMCell(dim)
        lstm_fw_cell = rnn_cell.MultiRNNCell([cell]*depth)
        lstm_bw_cell = rnn_cell.MultiRNNCell([cell]*depth)
        output, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, 
                                                           dtype=tf.float32)
        return output
        

def test():
    model = Model(16, 129, 1071, 1)

    print model.h1.get_shape()
    print model.h1_pool.get_shape()

    print model.h2.get_shape()
    print model.h2_pool.get_shape()

    print model.h3.get_shape()

    print model.h4.get_shape()

    print model.h6.get_shape()

    print model.scores.get_shape()
    return

if __name__ == "__main__":
    test()     
