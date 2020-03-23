# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 07:23:47 2020

@author: vedenev
"""

import tensorflow as tf
import cv2
import numpy as np
import time


FROZEN_GRAPH_FILENAME = './frozen_graph_2020_01_14.pb'
HEIGHT = 28
WIDTH = 28
N_INPUT_CHANNELS = 1

def prepare_input_data(x):
    return (np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1)).astype(np.float32)) / 255.0

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

x_val_4d = prepare_input_data(x_val)



# load frozen graph:
with tf.gfile.GFile(FROZEN_GRAPH_FILENAME, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="prefix")

input_tensor = graph.get_tensor_by_name('prefix/input:0')
output_tensor = graph.get_tensor_by_name('prefix/output:0')
batch_size = 32
with tf.Session(graph=graph) as sess:
    # warm up:
    for count in range(5):
        ind = count * batch_size
        prediction_tmp = sess.run(output_tensor, feed_dict={input_tensor: x_val_4d[ind: ind + batch_size, :, :, :]})
        
    correct = 0
    time_1 = time.time()
    for count in range(312):
        ind = count * batch_size
        prediction_tmp = sess.run(output_tensor, feed_dict={input_tensor: x_val_4d[ind: ind + batch_size, :, :, :]})
        prediction_label_tmp = np.argmax(prediction_tmp, axis=1)
        correct += np.where(prediction_label_tmp == y_val[ind: ind + batch_size])[0].size
    time_2 = time.time()
    time_from_cpu_mean = (time_2 - time_1) / (312 * 32)
    


accuracy = correct / (312 * 32)


print('accuracy =', accuracy)
print('time_from_cpu_mean =', time_from_cpu_mean)
#accuracy = 0.9928
#time_from_cpu_mean = 0.00207171847820282
