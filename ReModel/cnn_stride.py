import numpy as np
import tensorflow as tf
import os
import threading
import sys
import sys
import matplotlib.pyplot as plt
import time
import scipy.misc
from datetime import datetime

class Config():
    B, W, H, C = 36, 36, 36, 12
    train_step = 30000
    lr = 1e-3
    weight_decay = 0.005

    drop_out = 0.25

    Base_dir = r'G:\EstimatedCrop\Data\MODIS_City'
    # Base_dir = r'E:\estimate_corp'
    load_path = os.path.join(Base_dir, 'cnn_input')
    save_path = os.path.join(Base_dir, 'cnn_output')



def conv2d(input_data, out_channels, filter_size,stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        tf.summary.histogram('Weight',W)
        tf.summary.histogram('biases',b)
        return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b


def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")


def conv_relu_batch(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):#conv_relu_batch(self.x, 128, 3,1, name="conv1_1")
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        tf.summary.histogram('activations',r)
        return r

def dense(input_data, H, N=None, name="dense"):
    # regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/50000)
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, H])
        # reg_term = W.apply_regularization(regularizer)
        return tf.matmul(input_data, W, name="matmul") + b

def batch_normalization(input_data, axes=[0], name="batch"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

class NeuralModel():
    def __init__(self, config, name):

        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        # self.year = tf.placeholder(tf.float32, [None,1])
        # used for max image
        # self.image = tf.Variable(initial_value=init,name="image")

        # self.conv0_1 = conv_relu_batch(self.x, 32, 1,1, name="conv0_1")
        # # # conv1_1_d = tf.nn.dropout(self.conv1_1, self.keep_prob)
        # conv0_2 = conv_relu_batch(self.conv0_1, 32, 3,2, name="conv0_2")
        # # conv0_2_d = tf.nn.dropout(conv0_2, self.keep_prob)

        self.conv1_1 = conv_relu_batch(self.x, 64, 3,2, name="conv1_1")
        # conv1_1_d = tf.nn.dropout(self.conv1_1, self.keep_prob)
        conv1_2 = conv_relu_batch(self.conv1_1, 64, 3,1, name="conv1_2")
        # conv1_2_d = tf.nn.dropout(conv1_2, self.keep_prob)

        conv2_1 = conv_relu_batch(conv1_2, 128, 3,2, name="conv2_1")
        # conv2_1_d = tf.nn.dropout(conv2_1, self.keep_prob)
        conv2_2 = conv_relu_batch(conv2_1, 128, 3,2, name="conv2_2")
        # conv2_2_d = tf.nn.dropout(conv2_2, self.keep_prob)

        conv3_1 = conv_relu_batch(conv2_2, 256, 3,2, name="conv3_1")
        # conv3_1_d = tf.nn.dropout(conv3_1, self.keep_prob)
        conv3_2= conv_relu_batch(conv3_1, 256, 3,1, name="conv3_2")
        # conv3_2_d = tf.nn.dropout(conv3_2, self.keep_prob)
        conv3_3 = conv_relu_batch(conv3_2, 256, 3,2, name="conv3_3")
        conv3_3_d = tf.nn.dropout(conv3_3, self.keep_prob)

        dim = np.prod(conv3_3_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_3_d, [-1, dim])
        # flattened_d = tf.nn.dropout(flattened, 0.25)

        print (flattened.get_shape())
        self.fc6 = dense(flattened, 512, name="fc6")
        # self.fc6_r = tf.nn.relu(self.fc6)
        self.fc6_d = tf.nn.dropout(self.fc6, self.keep_prob)
        #
        #
        fc7 = dense(self.fc6_d, 1024, name="fc7")
        # fc7_r = tf.nn.relu(fc7)
        fc7_d = tf.nn.dropout(fc7, self.keep_prob)

        self.logits = tf.squeeze(dense(fc7_d, 1, name="dense"))
     
        # l2
        self.loss_err = tf.nn.l2_loss(self.logits - self.y)
        tf.summary.scalar('L2',self.loss_err)
        # l1
        # self.loss_err = tf.reduce_sum(tf.abs(self.logits - self.y))
        # average
        # self.loss_err = tf.abs(tf.reduce_sum(self.logits - self.y))

        regularizer = tf.contrib.layers.l2_regularizer(0.001)
        with tf.variable_scope('dense',regularizer=regularizer) as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')

        # lasso_param = tf.constant(0.9)
        # heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-50., tf.subtract(self.logits, lasso_param)))))
        # regularization_param = tf.multiply(heavyside_step, 99.)
        # self.loss = tf.add(tf.reduce_mean(tf.square(self.y - self.logits)), regularization_param)
        # tf.summary.scalar('lasso',self.loss)

        self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])*0.001
        self.loss = self.loss_err+self.loss_reg
        # self.loss = self.loss_err
        open('err.txt','a').write(str(self.loss)+',')
        # # learning rate decay
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.lr = tf.train.exponential_decay(config.lr_start, global_step,
        #                                            config.lr_decay_step, config.lr_decay_rate, staircase=False)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        #compute the accuracy
        # correct_prediction = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.y,1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

