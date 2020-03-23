# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 09:55:40 2019

@author: vedenev
"""

import tensorflow as tf
import numpy as np

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

SAVE_PATH = './weigths_1d'

#              c 3 x 3            p 2 x 2           c 5 x 5         p 3 x 3         c 3 x 3          c 1 x 1
# 28 x 28 x 1   ->   26 x 26 x 16   ->  13 x 13 x 16  ->  9 x 9 x 16 ->   3 x 3 x 16  ->  1 x 1 x 256  ->   1 x 1 x 10
# n mult:     97344                                 518400                          36864             2560
# 784                   36864           2704               1296              144                256            10

WEIGHTS_SAVE_PATH = './weights/best_model.ckpt'




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

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
# y_val.shape (10000,)
x_val = prepare_input_data(x_val) # (10000, 28, 28, 1)
x_val = x_val[:, :, :, 0] # (10000, 28, 28, 1)

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

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, WEIGHTS_SAVE_PATH)

W_conv1_value = sess.run(W_conv1)
# [filter_height, filter_width, in_channels, out_channels] -> [out_channels, in_channels, filter_height, filter_width]
# (3, 3, 1, 16) - > (16, 1, 3, 3)
W_conv1_1d = W_conv1_value.transpose((3, 2, 0, 1)).flatten()
print("W_conv1_1d.size =", W_conv1_1d.size)
W_conv1_1d.tofile(SAVE_PATH + '/' +'W_conv1.bin')
b_conv1_1d = sess.run(b_conv1)
b_conv1_1d.tofile(SAVE_PATH + '/' +'b_conv1.bin')
print("b_conv1_1d.size =", b_conv1_1d.size)

W_conv2_1d = sess.run(W_conv2).transpose((3, 2, 0, 1)).flatten()
W_conv2_1d.tofile(SAVE_PATH + '/' +'W_conv2.bin')
print("W_conv2_1d.size =", W_conv2_1d.size)
b_conv2_1d = sess.run(b_conv2)
b_conv2_1d.tofile(SAVE_PATH + '/' +'b_conv2.bin')
print("b_conv2_1d.size =", b_conv2_1d.size)

W_conv3_1d = sess.run(W_conv3).transpose((3, 2, 0, 1)).flatten()
W_conv3_1d.tofile(SAVE_PATH + '/' +'W_conv3.bin')
print("W_conv3_1d.size =", W_conv3_1d.size)
b_conv3_1d = sess.run(b_conv3)
b_conv3_1d.tofile(SAVE_PATH + '/' +'b_conv3.bin')
print("b_conv3_1d.size =", b_conv3_1d.size)

W_conv4_1d = sess.run(W_conv4).transpose((3, 2, 0, 1)).flatten()
W_conv4_1d.tofile(SAVE_PATH + '/' +'W_conv4.bin')
print("W_conv4_1d.size =", W_conv4_1d.size)
b_conv4_1d = sess.run(b_conv4)
b_conv4_1d.tofile(SAVE_PATH + '/' +'b_conv4.bin')
print("b_conv4_1d.size =", b_conv4_1d.size)

x_val_1d = x_val.flatten()
x_val.tofile(SAVE_PATH + '/' +'x_val.bin')
print("x_val_1d.size =", x_val_1d.size)

y_val.tofile(SAVE_PATH + '/' +'y_val.bin')
print("y_val.size =", y_val.size)