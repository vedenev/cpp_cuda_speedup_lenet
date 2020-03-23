# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:49:18 2019

@author: vedenev
"""


import numpy as np
import matplotlib.pyplot as plt 

LOSSES_SAVE_PATH = './losses'
SMOOTH_SIZE = 9

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


train_loss_per_epoch = np.load(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch.npy')
train_loss_per_epoch_epoch = np.load(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch_epoch.npy')
train_loss_per_epoch_calculated = np.load(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch_calculated.npy')
train_loss = np.load(LOSSES_SAVE_PATH + '/' + 'train_loss.npy')
train_loss_epoch = np.load(LOSSES_SAVE_PATH + '/' + 'train_loss_epoch.npy')
train_loss_epoch_calculated = np.load(LOSSES_SAVE_PATH + '/' + 'train_loss_epoch_calculated.npy')
val_loss_per_epoch = np.load(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch.npy')
val_accuracy_per_epoch = np.load(LOSSES_SAVE_PATH + '/' + 'val_accuracy_per_epoch.npy')
val_loss_per_epoch_epoch = np.load(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch_epoch.npy')
val_loss_per_epoch_calculated = np.load(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch_calculated.npy')

train_loss_per_epoch = train_loss_per_epoch[train_loss_per_epoch_calculated]
train_loss_per_epoch_epoch = train_loss_per_epoch_epoch[train_loss_per_epoch_calculated] + 0.5

train_loss = train_loss[train_loss_epoch_calculated]
train_loss_epoch = train_loss_epoch[train_loss_epoch_calculated]

val_loss_per_epoch = val_loss_per_epoch[val_loss_per_epoch_calculated]
val_accuracy_per_epoch = val_accuracy_per_epoch[val_loss_per_epoch_calculated]
val_loss_per_epoch_epoch = val_loss_per_epoch_epoch[val_loss_per_epoch_calculated]

val_accuracy_per_epoch_smooth = smooth(val_accuracy_per_epoch, SMOOTH_SIZE)

plt.subplot(2,1,1)
plt.plot(train_loss_epoch, train_loss, 'k.-', label='train steps')
plt.plot(train_loss_per_epoch_epoch, train_loss_per_epoch, 'r.-', label='train')
plt.plot(val_loss_per_epoch_epoch, val_loss_per_epoch, 'b.-', label='validation')
#plt.plot(fit_x, fit_y, 'y--', label='validation extrapolation')
plt.xlabel('epoch')
plt.ylabel('loss, MSE')
plt.ylim([0, np.max(train_loss)])
#plt.ylim([0, 1.5])
plt.legend()

plt.subplot(2,1,2)
plt.plot(val_loss_per_epoch_epoch, val_accuracy_per_epoch, 'b.-', label='validation')
plt.plot(val_loss_per_epoch_epoch, val_accuracy_per_epoch_smooth, 'r-', label='validation smoothed')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
