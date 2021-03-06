# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 07:23:47 2020

@author: vedenev
"""

import tensorflow as tf
import cv2
import numpy as np
import time


FROZEN_GRAPH_FILENAME = './frozen_graph_2020_01_14_tmp2.pb'
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
    for count in range(5):
        prediction_tmp = sess.run(output_tensor, feed_dict={index_tensor: count})
        

    time_1 = time.time()
    for count in range(78):
        prediction_tmp = sess.run(output_tensor, feed_dict={index_tensor: count})
    time_2 = time.time()
    time_from_gpu_mean = (time_2 - time_1) / (78 * 128)
    




print('time_from_gpu_mean =', time_from_gpu_mean)

# from CPU:
#accuracy = 0.9928
#time_from_cpu_mean = 0.00207171847820282
#time_from_cpu_mean = 0.0020748186588287355
#time_from_cpu_mean = 0.002072118520736694
#time_from_cpu_mean = 0.002102320241928101
#time_from_cpu_mean = 0.0020703184127807616

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

# from GPU no output batch:
# time_from_gpu_mean = 2.524182678033144e-05
# time_from_gpu_mean = 2.4240177411299486e-05
# time_from_gpu_mean = 2.293799741145892e-05
# time_from_gpu_mean = 2.3238504162201516e-05
# time_from_gpu_mean = 2.333868103913772e-05

# from GPU with output batch 32:
#time_from_cpu_mean = 7.141835223405789e-05
#time_from_cpu_mean = 6.971553636667055e-05
#time_from_cpu_mean = 7.081733873257271e-05
#time_from_cpu_mean = 7.06170088587663e-05
#time_from_cpu_mean = 7.63264699624135e-05

# from GPU with output batch 64:
# time_from_cpu_mean = 4.527502908156468e-05
# time_from_cpu_mean = 3.9465391100981295e-05
# time_from_cpu_mean = 4.026671059620686e-05
# time_from_cpu_mean = 3.9265061227174905e-05
# time_from_cpu_mean = 3.9064731353368515e-05

# from GPU with output batch 128:
#accuracy = 0.9927884615384616
#time_from_cpu_mean = 2.7445479272267756e-05
#
#accuracy = 0.9927884615384616
#time_from_cpu_mean = 2.8146645770623134e-05
#
#accuracy = 0.9927884615384616
#time_from_cpu_mean = 2.7946315896816744e-05


# from GPU with output batch 256:
#accuracy = 0.9927884615384616
#time_from_cpu_mean = 2.2537361543912153e-05
#
#accuracy = 0.9927884615384616
#time_from_cpu_mean = 2.1235181544071587e-05
#
#accuracy = 0.9927884615384616
#time_from_cpu_mean = 2.1135004667135385e-05
#
#accuracy = 0.9927884615384616
#time_from_cpu_mean = 2.1235205424137606e-05
#
#accuracy = 0.9927884615384616
#time_from_cpu_mean = 2.1235181544071587e-05