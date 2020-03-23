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
        

    time_1 = time.time()
    for count in range(y_val.size):
        prediction_tmp = sess.run(output_tensor, feed_dict={index_tensor: count})
    time_2 = time.time()
    time_from_gpu_mean = (time_2 - time_1) / y_val.size
    




print('time_from_gpu_mean =', time_from_gpu_mean)

# from CPU:
#accuracy = 0.9928
#time_from_cpu_mean = 0.00207171847820282

#from GPU:
#accuracy = 0.9928
# run1) time_from_gpu_mean = 0.0019116093397140504
# run2) time_from_gpu_mean = 0.0019091092109680176
# run3) time_from_gpu_mean = 0.0018981085777282714
# run4) time_from_gpu_mean = 0.0019231100082397461
# run5) time_from_gpu_mean = 0.0019405109882354736

# from GPU no output:
# run1) time_from_gpu_mean = 0.0019041089057922364
# run2) time_from_gpu_mean = 0.001856806206703186
# run3) time_from_gpu_mean = 0.0017731014251708985
# run4) time_from_gpu_mean = 0.001758500576019287
# run5) time_from_gpu_mean = 0.001875207257270813
