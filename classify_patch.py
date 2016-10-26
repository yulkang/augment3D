#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:09:43 2016

@author: yulkang
"""

#%% Import
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import pandas as pd
import warnings
import os
from PIL import Image
import matplotlib.pyplot as plt
import time

from pysy import zipPickle
import compare_cand_vs_annot as annot
import import_mhd as mhd

#%% Choose train and test datasets
# Train: All positive in the subset + 3x negative (random subset)
cands_pos = annot.cands_pos
cands_neg = annot.cands_neg

subset_incl_pos = np.arange(9)
subset_incl_neg = np.array([0])

scale_incl = 0
fmt = mhd.output_formats.iloc[scale_incl,:]
scale_incl_pos = (cands_pos.loc[:,'radius'] >= fmt['radius_min_incl']) \
               & (cands_pos.loc[:,'radius'] < fmt['radius_max_incl'])

cands_pos = cands_pos.ix[cands_pos.subset.isin(subset_incl_pos) \
                         & scale_incl_pos,:]
cands_neg = cands_neg.ix[cands_neg.subset.isin(subset_incl_neg),:]

n_pos = len(cands_pos)
memory_available_MB = 1000
memory_per_cand_neg_MB = 0.5
n_neg = np.int32(np.min((memory_available_MB / memory_per_cand_neg_MB, 
                len(cands_neg))))

cands_neg = cands_neg.iloc[:n_neg,:]
cands_all = pd.concat((cands_pos, cands_neg), axis=0)
n_all = len(cands_all)

#%% Load dataset
def load_cand(cands, scale=0):
    
    t_st = time.time()
    
    n_cand = len(cands)
    n_loaded = 0
    print('Loading %d candidates' % n_cand)
    
    ix_loaded = np.zeros((n_cand), dtype=np.int32)
    img_all = None
    
    for i_cand in range(n_cand):
        cand = cands.iloc[i_cand,:]
        patch_file, _, _ = mhd.cand_scale2patch_file(
                cand, 
                output_format=mhd.output_formats.iloc[0,:])
        if os.path.isfile(patch_file + '.zpkl'):
            L = zipPickle.load(patch_file + '.zpkl')
            n_loaded += 1
        else:
            continue
        
        if n_loaded == 1:
            siz = np.concatenate(([n_cand], L['img'].shape))
            img_all = np.zeros(siz, dtype=np.float32)
            siz1 = siz.copy()
            siz1[0] = 1
            print('Loading images of size:')
            print(L['img'].shape)
            
        try:
            img_all[n_loaded-1,:,:,:] = np.reshape(L['img'], siz1)
        except ValueError as err:
            n_loaded -= 1
            warnings.warn(err.message)
            print('Potential shape discrepancy:')
            print(L['img'].shape)
            continue
            
        ix_loaded[n_loaded-1] = i_cand
            
    if img_all is not None:
        img_all = img_all[:n_loaded,:,:,:]
        img_all = img_all.reshape(np.concatenate((siz,[1])))
        ix_loaded = ix_loaded[:n_loaded]
        ix_loaded = cands.index[ix_loaded]
        label_all = cands.loc[ix_loaded,'is_pos']
    else:
        label_all = None
        ix_loaded = None

    t_el = time.time() - t_st
    print('Time elapsed: %1.2f sec / %d img = %1.2f msec/img' % 
          (t_el, img_all.shape[0], t_el / img_all.shape[0] * 1e3))
        
    print('Last image loaded:')
    print(np.int32(ix_loaded[-1])) # DEBUG    
    patch1 = img_all[-1,img_all.shape[1]/2,:,:,0]
    plt.imshow(patch1, cmap='gray')
    plt.show()
    
    return img_all, label_all, ix_loaded

img_neg, label_neg, ix_loaded_neg = load_cand(cands_neg.iloc[:,:])
img_pos, label_pos, ix_loaded_pos = load_cand(cands_pos.iloc[:,:])

#%%
if 'sess' in vars():
    sess.close()
    
sess = tf.InteractiveSession()

#%% Settings
img_size = img_neg.shape[1]

n_labels = 2
n_chan = 1 # grayscale
batch_size = 128 # 16
patch_size = 4
stride_size = 2
pool_size = 2
depth_conv = 64
n_hidden = 64
  
# Input data.
x = tf.placeholder(tf.float32, 
                   shape=[None, img_size, img_size, img_size, n_chan])
y = tf.placeholder(tf.float32, 
                   shape=[None, n_labels])

#tf_train_dataset = tf.placeholder(
#  tf.float32, shape=(batch_size, img_size, img_size, n_chan))
#tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_labels))
#tf_valid_dataset = tf.constant(valid_dataset)
#tf_test_dataset = tf.constant(test_dataset)

#%% Variables.
keep_prob_train = [1, 1, 1, .5, .5]
keep_all = np.ones(len(keep_prob_train))

layer_kind = ['conv', 'conv', 'conv', 'dense', 'dense']
is_conv = np.array([kind1 == 'conv' for kind1 in layer_kind], dtype=np.int32)
width_mult = is_conv * patch_size
stride_mult = is_conv * stride_size
width_pool = np.array([0, 0, 0, 0, 0]) * pool_size
depth_out = np.ones(len(layer_kind)) * depth_conv
depth_in = [n_chan] + list(depth_out[:-1])

weights0 = list()
biases = list()

#%%
n_layer = len(layer_kind)
for layer in range(n_layer):
    if layer_kind[layer] == 'conv':
        weights0.append(tf.Variable(tf.truncated_normal(
            [width_mult[layer], 
             width_mult[layer], 
             width_mult[layer], 
             depth_in[layer], 
             depth_out[layer]],
            stddev = 0.1)))
        biases.append(tf.Variable(tf.constant(1.0, shape=[depth_out[layer]])))
    elif layer_kind[layer] == 'dense':
        pass
    else:
        raise ValueError('layer_kind[%d]=%s not allowed!' % 
                         (layer, layer_kind[layer]))

#%%
weights0.append(tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, patch_size, n_chan, depth_conv[0]], stddev=0.1)))
biases.append(tf.Variable(tf.zeros([depth_conv[0]])))

weights0.append(tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_conv[0], depth_conv[1]], stddev=0.1)))
biases.append(tf.Variable(tf.constant(1.0, shape=[depth])))

weights0.append(tf.Variable(tf.truncated_normal(
    [img_size // stride_mult[0]**2 * img_size // stride_mult[0]**2 * depth, 
     n_hidden], stddev=0.1)))
biases.append(tf.Variable(tf.constant(1.0, shape=[n_hidden])))

weights0.append(tf.Variable(tf.truncated_normal(
    [n_hidden, n_labels], stddev=0.1)))
biases.append(tf.Variable(tf.constant(1.0, shape=[n_labels])))

#%% Model.
def accuracy(predictions, labels):
    return (100.0 * np.mean(predictions == labels))
  
def add_conv2d(inp, weight, bias,
             keep_prob = 1, 
             stride = 2):
  
    weight_dropout = tf.nn.dropout(weight, keep_prob)
    stride_model = [1, stride, stride, 1]
    conv = tf.nn.conv2d(inp, weight_dropout, stride_model, 
                      padding='VALID')
    hidden = tf.nn.relu(conv + bias)
    hidden = tf.nn.max_pool(hidden, stride_model, stride_model, 
                          padding='VALID')
    return hidden
  
def add_dense(inp, weight, bias, keep_prob = 1, depth, is_final = False):
    shape = np.array(inp.get_shape().as_list())
    reshape = tf.reshape(inp, [-1, np.prod(shape[1:])])
    dense = tf.matmul(reshape, weight) + bias
    if not is_final:
        dense = tf.nn.relu(dense)
    return dense
  
def model(keep_prob):
    weights = list()
    for layer in range(len(weights0)):
        weights.append(tf.nn.dropout(weights0[layer], keep_prob[layer]))
  
    layer = 0
    hidden = add_conv2d(x, weights[0], keep_prob[0], stride[0])
  
    for i_conv in np.arange(1, n_conv):
        layer += 1
        hidden = add_conv2d(hidden, weights[layer], keep_prob[layer], 
                        stride_mult[layer])
  
    for i_dense in np.arange(1, n_dense):
        layer += 1
        hidden = add_dense(hidden, weights[layer], bias[layer], keep_prob[layer], 
                       is_final = i_dense == n_dense)
    
    return hidden

# Training computation / Optimizer
logits_train = model(keep_prob_train)
loss_train = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(logits_train, y))
train_prediction = tf.nn.softmax(logits_train)

logits = model(keep_all)
valid_prediction = tf.nn.softmax(logits)  

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss_train)

#%%
sess.run(tf.initialize_all_variables())
print('Initialized')
step = 0

#%%
num_steps = 1001 # 3000 steps give test accuracy 91.5%
display_per_step = 200

#%%
for step1 in range(num_steps):
  step += 1
  
  offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
  batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
  batch_labels = train_labels[offset:(offset + batch_size), :]
  feed_dict = {x : batch_data, y : batch_labels}
  _, l, predictions = sess.run(
    [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
  if (step % display_per_step == 0):
    print('Minibatch loss at step %d: %f' % (step, l))
    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    print('Validation accuracy: %.1f%%' % accuracy(
      valid_prediction.eval(
        feed_dict = {x : valid_dataset}), 
        valid_labels))
    
#%%
print('Test accuracy: %.1f%%' % accuracy(
  valid_prediction.eval(
    feed_dict = {x : test_dataset}), test_labels))
