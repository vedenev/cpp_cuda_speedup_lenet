# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 07:23:47 2020

@author: vedenev
"""

import tensorflow as tf
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
batch_size_all = np.asarray([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072])
times = np.zeros(batch_size_all.size, np.float32)
n_repeats = 5
for batch_size_count in range(batch_size_all.size):
    print(batch_size_count, batch_size_all.size - 1)
    batch_size = batch_size_all[batch_size_count]
    n_batches = x_val_4d.shape[0] // batch_size
    daset_size_1 = batch_size * n_batches
    with tf.Session(graph=graph) as sess:
        
        time_from_cpu_mean_all = np.zeros(n_repeats, np.float32)
        
        # warm up:
        for count in range(3):
            ind = count * batch_size
            prediction_tmp = sess.run(output_tensor, feed_dict={input_tensor: x_val_4d[ind: ind + batch_size, :, :, :]})
        
        for repeat_count in range(n_repeats):
            correct = 0
            time_1 = time.time()
            for count in range(n_batches):
                ind = count * batch_size
                prediction_tmp = sess.run(output_tensor, feed_dict={input_tensor: x_val_4d[ind: ind + batch_size, :, :, :]})
                prediction_label_tmp = np.argmax(prediction_tmp, axis=1)
                correct += np.where(prediction_label_tmp == y_val[ind: ind + batch_size])[0].size
            time_2 = time.time()
            time_from_cpu_mean = (time_2 - time_1) / daset_size_1
            time_from_cpu_mean_all[repeat_count] = time_from_cpu_mean
        time_from_cpu_mean_mean = np.mean(time_from_cpu_mean_all)
        times[batch_size_count] = time_from_cpu_mean_mean
    



# batch_size_all = np.asarray([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
#times = array([2.0778987e-03, 1.0584205e-03, 5.3335051e-04, 2.6573520e-04,
#       1.3936797e-04, 6.9875794e-05, 3.9305130e-05, 2.7124948e-05,
#       2.0554060e-05, 1.6653910e-05], dtype=float32)

np.save('batch_size_all_tf_1_15_k80__2020_03_23.npy', batch_size_all)
np.save('times_tf_1_15_k80__2020_03_23.npy', times)