# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 09:55:40 2019

@author: vedenev
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

HEIGHT = 28
WIDTH = 28
N_INPUT_CHANNELS = 1
N_CLASSES = 10

# net architecture: cpcpff
N_FEATUREMAPS =   [16, 16, 256]
KERNELS_X_SIZE = [3, 5, 3, 1]
KERNELS_Y_SIZE = [3, 5, 3, 1]
POOLING_X_SIZE = [2, 3]
POOLING_Y_SIZE = [2, 3]
DROPOUT_RATE = 0.5

#              c 3 x 3            p 2 x 2           c 5 x 5         p 3 x 3         c 3 x 3          c 1 x 1
# 28 x 28 x 1   ->   26 x 26 x 16   ->  13 x 13 x 16  ->  9 x 9 x 16 ->   3 x 3 x 16  ->  1 x 1 x 256  ->   1 x 1 x 10
# n mult:     97344                                 518400                          36864             2560
# 784                   36864           2704               1296              144                256            10

WEIGHTS_SAVE_PATH = './weights/best_model.ckpt'
FROZEN_GRAPH_FILENAME = './frozen_graph_2020_01_14.pb'

def weight_variable(name, shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    #initial = 0.01*np.ones(shape, dtype=np.float32)
    #return tf.Variable(initial)
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    
def bias_variable(name, shape):
    #initial = tf.constant(0.001, shape=shape)
    #initial = tf.constant(0.0, shape=shape)
    #return tf.Variable(initial)
    return tf.get_variable(name, shape=shape, initializer=tf.zeros_initializer())

def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

def max_pool(x, size_x, size_y):
    return tf.nn.max_pool(x, ksize=[1,size_x, size_y,1], strides=[1,size_x, size_y,1], padding='VALID')

def prepare_input_data(x):
    return (np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1)).astype(np.float32)) / 255.0

def labels_to_one_hots(y):
    y_one_hot = np.zeros((y.size, N_CLASSES), np.float32)
    for sample_count in range(y.size):
        label = y[sample_count]
        y_one_hot[sample_count, label] = 1.0
    return y_one_hot
# net architecture: ccpccpff
x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, N_INPUT_CHANNELS], name='input')


W_conv1 = weight_variable("W_conv1", [KERNELS_X_SIZE[0], KERNELS_Y_SIZE[0], N_INPUT_CHANNELS, N_FEATUREMAPS[0]])
b_conv1 = bias_variable("b_conv1", [N_FEATUREMAPS[0]])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

h_pool1 = max_pool(h_conv1, POOLING_X_SIZE[0], POOLING_Y_SIZE[0])

W_conv2 = weight_variable("W_conv2", [KERNELS_X_SIZE[1], KERNELS_Y_SIZE[1], N_FEATUREMAPS[0], N_FEATUREMAPS[1]])
b_conv2 = bias_variable("b_conv2", [N_FEATUREMAPS[1]])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool(h_conv2, POOLING_X_SIZE[1], POOLING_Y_SIZE[1])

W_conv3 = weight_variable("W_conv3", [KERNELS_X_SIZE[2], KERNELS_Y_SIZE[2], N_FEATUREMAPS[1], N_FEATUREMAPS[2]])
b_conv3 = bias_variable("b_conv3", [N_FEATUREMAPS[2]])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_conv4 = weight_variable("W_conv4", [KERNELS_X_SIZE[3], KERNELS_Y_SIZE[3], N_FEATUREMAPS[2], N_CLASSES])
b_conv4 = bias_variable("b_conv4", [N_CLASSES])
h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4

flattened4 = h_conv4[:, 0, 0, :]

output = tf.nn.softmax(flattened4, name='output')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, WEIGHTS_SAVE_PATH)

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train = prepare_input_data(x_train)
y_train = labels_to_one_hots(y_train)
x_val = prepare_input_data(x_val)
y_val = labels_to_one_hots(y_val)

sample_index = 0
h_conv1_value = sess.run(h_conv1, feed_dict={x: x_val[sample_index: sample_index + 1, :, :, :]})
W_conv1_value = sess.run(W_conv1, feed_dict={x: x_val[sample_index: sample_index + 1, :, :, :]})
h_pool1_value = sess.run(h_pool1, feed_dict={x: x_val[sample_index: sample_index + 1, :, :, :]})
h_conv2_value = sess.run(h_conv2, feed_dict={x: x_val[sample_index: sample_index + 1, :, :, :]})
h_pool2_value = sess.run(h_pool2, feed_dict={x: x_val[sample_index: sample_index + 1, :, :, :]})

#sess.close()
