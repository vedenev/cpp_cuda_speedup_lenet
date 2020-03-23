# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 07:23:47 2020

@author: vedenev
"""

import tensorflow as tf
import cv2
import numpy as np
import time


FROZEN_GRAPH_FILENAME = './frozen_graph_2020_01_14_tmp.pb'
HEIGHT = 28
WIDTH = 28
N_INPUT_CHANNELS = 1


(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# load frozen graph:
with tf.gfile.GFile(FROZEN_GRAPH_FILENAME, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="prefix")

index_tensor = graph.get_tensor_by_name('prefix/index:0')
output_tensor = graph.get_tensor_by_name('prefix/output:0')

with tf.Session(graph=graph) as sess:
    # warm up:
    for count in range(10):
        prediction_tmp = sess.run(output_tensor, feed_dict={index_tensor: count})
        
    correct = 0
    time_1 = time.time()
    for count in range(y_val.size):
        prediction_tmp = sess.run(output_tensor, feed_dict={index_tensor: count})
        prediction_label_tmp = np.argmax(prediction_tmp[0, :])
        if prediction_label_tmp == y_val[count]:
            correct += 1
    time_2 = time.time()
    time_from_gpu_mean = (time_2 - time_1) / y_val.size
    


accuracy = correct / y_val.size


print('accuracy =', accuracy)
print('time_from_gpu_mean =', time_from_gpu_mean)

# from CPU:
#accuracy = 0.9928
#time_from_cpu_mean = 0.00207171847820282

#from GPU:
#accuracy = 0.9928
#time_from_gpu_mean = 0.0019116093397140504
