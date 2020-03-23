# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:46:45 2020

@author: vedenev
"""

import numpy as np
from scipy.signal import convolve2d

test_features_input = np.random.randn(3, 28, 28).astype(np.float32)
test_weights = np.random.randn(10, 3, 3, 3).astype(np.float32)
test_beases = np.random.randn(10).astype(np.float32)

test_features_output = np.zeros((10, 26, 26), np.float32)
for out_count in range(10):
    for inp_count in range(3):
        weights_tmp1 = test_weights[out_count, inp_count, :, :]
        weights_tmp2 = np.flipud(np.fliplr(weights_tmp1))
        test_features_output[out_count, :, :] += convolve2d(test_features_input[inp_count, :, :], weights_tmp2, mode='valid')
    test_features_output[out_count, :, :] += test_beases[out_count]
    

test_features_input_1d = test_features_input.flatten()
test_features_input_1d.tofile('test_features_input_1d.bin')

test_weights_1d = test_weights.flatten()
test_weights_1d.tofile('test_weights_1d.bin')

test_beases_1d = test_beases.flatten()
test_beases_1d.tofile('test_beases_1d.bin')

test_features_output_1d = test_features_output.flatten()
test_features_output_1d.tofile('test_features_output_1d.bin')