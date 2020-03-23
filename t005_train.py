# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:59:08 2020

@author: vedenev
"""

import tensorflow as tf
import datetime
import numpy as np
import os
import glob

N_EPOCHS = 200
LEARNING_RATE = 0.0001
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL = 128
VAL_STEP_RELATIVE = 4
PRINT_STEP_TRAIN = 50
PRINT_STEP_VAL = 50
LOSS_SAVE_STEP = 10
WEIGHTS_SAVE_PATH = './weights'
LOSSES_SAVE_PATH = './losses'

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

#              c 3 x 3            p 2 x 2           c 5 x 5         p 3 x 3         c 3 x 3          c 1 x 1
# 28 x 28 x 1   ->   26 x 26 x 16   ->  13 x 13 x 16  ->  9 x 9 x 16 ->   3 x 3 x 16  ->  1 x 1 x 256  ->   1 x 1 x 10
# n mult:     97344                                 518400                          36864             2560
# 784                   36864           2704               1296              144                256            10


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

def labels_to_one_hots(y):
    y_one_hot = np.zeros((y.size, N_CLASSES), np.float32)
    for sample_count in range(y.size):
        label = y[sample_count]
        y_one_hot[sample_count, label] = 1.0
    return y_one_hot

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train = prepare_input_data(x_train)
y_train = labels_to_one_hots(y_train)
x_val = prepare_input_data(x_val)
y_val = labels_to_one_hots(y_val)

t1970 = datetime.datetime(1970,1,1,0,0,0)
seed = int(np.round((datetime.datetime.now()- t1970).total_seconds())*100.0)
tf.set_random_seed(seed)

# net architecture: ccpccpff
x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, N_INPUT_CHANNELS])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
dropout_switch = tf.placeholder(tf.bool)

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

dropout = tf.layers.dropout(h_conv3, rate=DROPOUT_RATE, training=dropout_switch)

W_conv4 = weight_variable("W_conv4", [KERNELS_X_SIZE[3], KERNELS_Y_SIZE[3], N_FEATUREMAPS[2], N_CLASSES])
b_conv4 = bias_variable("b_conv4", [N_CLASSES])
h_conv4 = conv2d(dropout, W_conv4) + b_conv4

flattened4 = h_conv4[:, 0, 0, :]

output = tf.nn.softmax(flattened4)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=flattened4, labels=y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
train_step = optimizer.minimize(loss)

train_size = x_train.shape[0]
val_size = x_val.shape[0]

print('train_size =', train_size)
print('val_size =', val_size)

train_n_batches = train_size // BATCH_SIZE_TRAIN
val_n_batches = val_size // BATCH_SIZE_VAL

val_step = train_n_batches // VAL_STEP_RELATIVE

# val loss array precaluslte size:
val_loss_per_epoch_size = 0
for epoch_count in range(N_EPOCHS):
    for train_batch_count in range(train_n_batches):
        step_count = epoch_count * train_n_batches + train_batch_count
        if step_count % val_step == 0:
            val_loss_per_epoch_size += 1

print("val_loss_per_epoch_size =", val_loss_per_epoch_size)

train_loss_per_epoch = np.zeros(N_EPOCHS, np.float32)
train_loss_per_epoch_epoch = np.zeros(N_EPOCHS, np.float32)
train_loss_per_epoch_calculated = np.zeros(N_EPOCHS, np.bool)
n_train_steps = train_n_batches * N_EPOCHS
print("n_train_steps =", n_train_steps)
train_loss = np.zeros(n_train_steps, np.float32)
train_loss_epoch = np.zeros(n_train_steps, np.float32)
train_loss_epoch_calculated = np.zeros(n_train_steps, np.bool)
val_loss_per_epoch = np.zeros(val_loss_per_epoch_size, np.float32)
val_accuracy_per_epoch = np.zeros(val_loss_per_epoch_size, np.float32)
val_loss_per_epoch_epoch = np.zeros(val_loss_per_epoch_size, np.float32)
val_loss_per_epoch_calculated = np.zeros(val_loss_per_epoch_size, np.bool)
val_loss_per_epoch_count = 0
val_accuracy_max = -1.0

sess = tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
step_count = 0
for epoch_count in range(N_EPOCHS):
    # train:
    train_loss_sum_t = 0.0
    for train_batch_count in range(train_n_batches):
        step_count = epoch_count * train_n_batches + train_batch_count
        epoch_float = float(step_count) / train_n_batches
        data_ind_1 = train_batch_count * BATCH_SIZE_TRAIN
        data_ind_2 = data_ind_1 + BATCH_SIZE_TRAIN
        image_t = x_train[data_ind_1:data_ind_2, :, :, :]
        one_hot_t = y_train[data_ind_1:data_ind_2, :]
        _, train_loss_t = sess.run([train_step, loss], feed_dict={x: image_t, y: one_hot_t, dropout_switch:True})
        if step_count % PRINT_STEP_TRAIN == 0:
            print('epoch =', epoch_count, ' train_batch_count =',train_batch_count,  ' train_loss_t =', train_loss_t)
        train_loss_sum_t += train_loss_t
        train_loss[step_count] = train_loss_t
        train_loss_epoch[step_count] = epoch_float
        train_loss_epoch_calculated[step_count] = True
        
        if step_count % val_step == 0:
            # validate:
            val_loss_sum_t = 0.0
            val_accuracy_sum_t = 0.0
            
            for val_batch_count in range(val_n_batches):
                data_val_ind_1 = val_batch_count * BATCH_SIZE_VAL
                data_val_ind_2 = data_val_ind_1 + BATCH_SIZE_VAL
                image_t = x_val[data_val_ind_1:data_val_ind_2, :, :, :]
                one_hot_t = y_val[data_val_ind_1:data_val_ind_2, :]
                val_loss_t, prediction_t = sess.run([loss, output], feed_dict={x: image_t, y: one_hot_t, dropout_switch:False})
                prediction_labels_t = np.argmax(prediction_t, axis=1)
                labels_t = np.argmax(one_hot_t, axis=1)
                val_accuracy_t = np.mean(prediction_labels_t == labels_t)
                if val_batch_count % PRINT_STEP_VAL == 0:
                    print('epoch =', epoch_count, ' val_batch_count =', val_batch_count,  ' val_loss_t =', val_loss_t)
                val_loss_sum_t += val_loss_t
                val_accuracy_sum_t += val_accuracy_t
                
            val_loss_t = val_loss_sum_t / val_n_batches
            val_accuracy_t = val_accuracy_sum_t / val_n_batches
            val_loss_per_epoch[val_loss_per_epoch_count] = val_loss_t
            val_accuracy_per_epoch[val_loss_per_epoch_count] = val_accuracy_t
            val_loss_per_epoch_epoch[val_loss_per_epoch_count] = epoch_float
            val_loss_per_epoch_calculated[val_loss_per_epoch_count] = True
            val_loss_per_epoch_count += 1
            
            if val_accuracy_t > val_accuracy_max:
                val_accuracy_max = val_accuracy_t
                saver.save(sess, WEIGHTS_SAVE_PATH + '/' + 'best_model.ckpt')
                
                # wtite epoch and loss to filenames of txt files:
                to_remove = glob.glob(WEIGHTS_SAVE_PATH + '/*.txt')
                for file_count in range(len(to_remove)):
                    os.remove(to_remove[file_count])
                text_file = open(WEIGHTS_SAVE_PATH + '/' + 'ep_' + str(epoch_count) + '.txt', 'w')
                text_file.write('epoch: ' + str(epoch_count))
                text_file.close()
                val_loss_min_str = str(val_accuracy_max).replace('.', '_')
                text_file = open(WEIGHTS_SAVE_PATH + '/' + 'val_accuracy_max__' + val_loss_min_str + '.txt', 'w')
                text_file.write('val_accuracy_max: ' + str(val_accuracy_max))
                text_file.close()
            
        
        if step_count % LOSS_SAVE_STEP == 0:
            np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch.npy', train_loss_per_epoch)
            np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch_epoch.npy', train_loss_per_epoch_epoch)
            np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch_calculated.npy', train_loss_per_epoch_calculated)
            np.save(LOSSES_SAVE_PATH + '/' + 'train_loss.npy', train_loss)
            np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_epoch.npy', train_loss_epoch)
            np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_epoch_calculated.npy', train_loss_epoch_calculated)
            np.save(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch.npy', val_loss_per_epoch)
            np.save(LOSSES_SAVE_PATH + '/' + 'val_accuracy_per_epoch.npy', val_accuracy_per_epoch)
            np.save(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch_epoch.npy', val_loss_per_epoch_epoch)
            np.save(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch_calculated.npy', val_loss_per_epoch_calculated)
        
        step_count += 1
        
    train_loss_per_epoch[epoch_count] = train_loss_sum_t / train_n_batches
    train_loss_per_epoch_epoch[epoch_count] = epoch_count
    train_loss_per_epoch_calculated[epoch_count] = True
    
    
np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch.npy', train_loss_per_epoch)
np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch_epoch.npy', train_loss_per_epoch_epoch)
np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_per_epoch_calculated.npy', train_loss_per_epoch_calculated)
np.save(LOSSES_SAVE_PATH + '/' + 'train_loss.npy', train_loss)
np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_epoch.npy', train_loss_epoch)
np.save(LOSSES_SAVE_PATH + '/' + 'train_loss_epoch_calculated.npy', train_loss_epoch_calculated)
np.save(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch.npy', val_loss_per_epoch)
np.save(LOSSES_SAVE_PATH + '/' + 'val_accuracy_per_epoch.npy', val_accuracy_per_epoch)
np.save(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch_epoch.npy', val_loss_per_epoch_epoch)
np.save(LOSSES_SAVE_PATH + '/' + 'val_loss_per_epoch_calculated.npy', val_loss_per_epoch_calculated)

sess.close()