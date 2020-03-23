# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:42:38 2020

@author: vedenev
"""

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#x_train.shape = (60000, 28, 28)
# type(x_train[0, 0, 0]) # numpy.uint8

#y_train.shape =  (60000,)
# y_train = array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)


# x_test.shape = (10000, 28, 28)
# y_test.shape = (10000,)

