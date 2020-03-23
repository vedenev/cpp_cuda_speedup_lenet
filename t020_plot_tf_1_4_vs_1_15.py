# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 07:52:49 2020

@author: vedenev
"""
import numpy as np
import matplotlib.pyplot as plt

batch_size_all_tf_1_4 = np.load('batch_size_all_tf_1_4_k80__2020_03_23.npy')
times_tf_1_4 = np.load('times_tf_1_4_k80__2020_03_23.npy')

batch_size_all_tf_1_15 = np.load('batch_size_all_tf_1_15_k80__2020_03_23.npy')
times_tf_1_15 = np.load('times_tf_1_15_k80__2020_03_23.npy')




plt.semilogy(batch_size_all_tf_1_4, times_tf_1_4, 'k.-', label='tf1.4 k80')
plt.semilogy(batch_size_all_tf_1_15, times_tf_1_15, 'g.-', label='tf1.15 k80')
#plt.semilogy([1], [0.00104740], 'gx', label='cuda code optimized batch_size=1')
#plt.semilogy(cuda_codes_times[:, 0], cuda_codes_times[:, 1], 'r.-', label='cuda codes')
plt.xlabel('batch size')
plt.ylabel('mean time per sample, sec')
plt.legend()