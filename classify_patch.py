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

#%% Functions to build the network
def accuracy(predictions, labels):
    return (100.0 * np.mean(predictions == labels))
  
def add_conv(inp, 
             width_mult, 
             stride_mult,
             depth_out,
             width_pool = 0,
             stride_pool = 2,
             stddev = 0.1,
             bias = 1.0,
             keep_prob = 1):
    
    depth_in = inp.get_shape().as_list()[-1]
    weight = tf.Variable(tf.truncated_normal(
            [width_mult] * 3 + [depth_in, depth_out],
            stddev = stddev))
    weight_dropout = tf.nn.dropout(weight, keep_prob)
    conv_op = tf.nn.conv3d(inp, weight_dropout, 
                           [1] + [stride_mult] * 3 + [1], 
                           padding='VALID')
    bias_op = tf.Variable(tf.constant(bias, shape=[depth_out]))
    hidden = tf.nn.relu(conv_op + bias_op)
    
    if width_pool > 0:
        hidden = tf.nn.max_pool3d(hidden, 
                                  [1] + [width_pool] * 3 + [1], 
                                  [1] + [stride_pool] * 3 + [1], 
                                  padding='VALID')
    return hidden
  
def add_dense(inp, depth_out, 
              bias = 1.0,
              keep_prob = 1,
              stddev = 0.1, 
              is_final = False):
    shape_inp = np.array(inp.get_shape().as_list())
    print('  shape_inp:', end='')
    print(shape_inp)
    n_inp = np.prod(shape_inp[1:])
    print('  n_inp: %d' % n_inp)
    inp_reshaped = tf.reshape(inp, [-1, n_inp])
    weight = tf.Variable(tf.truncated_normal(
            [n_inp, depth_out], stddev = stddev))    
    weight_dropout = tf.nn.dropout(weight, keep_prob)
    print('  weight:', end='')
    print(weight_dropout.get_shape().as_list())
    bias_op = tf.Variable(tf.constant(bias), [depth_out])
    dense = tf.matmul(inp_reshaped, weight_dropout) + bias_op
    if not is_final:
        dense = tf.nn.relu(dense)
    return dense

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

# Variables.
keep_probs_train = [1, 1, 1, .5, .5]
keep_probs_all = np.ones(len(keep_probs_train))

layer_kind = ['conv', 'conv', 'conv', 'dense', 'dense']
is_layer_kind = lambda kind: \
    np.array([kind1 == kind for kind1 in layer_kind], 
             dtype=np.int32)
n_layer = len(layer_kind)

is_conv = is_layer_kind('conv')
is_dense = is_layer_kind('dense')

widths_mult = is_conv * patch_size
strides_mult = is_conv * stride_size

widths_pool = np.zeros(n_layer, dtype=np.int32) * pool_size
strides_pool = np.zeros(n_layer, dtype=np.int32) * pool_size

depth_out = np.ones(n_layer, dtype=np.int32) * depth_conv
depth_out[-1] = n_labels

weights0 = list()
biases = list()

def model(keep_probs):
    curr_layer = x
    for layer in range(n_layer):
        print('--- Layer %d ---' % layer)
        print('Input :', end='')
        print(curr_layer.get_shape().as_list(), end='\n')
        
        if layer_kind[layer] == 'conv':
            curr_layer = add_conv(curr_layer,
                                  widths_mult[layer],
                                  strides_mult[layer],
                                  depth_out[layer],
                                  width_pool=widths_pool[layer],
                                  stride_pool=strides_pool[layer],
                                  keep_prob=keep_probs[layer])
            
        elif layer_kind[layer] == 'dense':
            curr_layer = add_dense(curr_layer,
                                   depth_out[layer],
                                   keep_prob=keep_probs[layer],
                                   is_final=layer >= n_layer - 1)
        else:
            raise ValueError('layer_kind[%d]=%s not allowed!' % 
                             (layer, layer_kind[layer]))
        
        print('Output:', end='')
        print(curr_layer.get_shape().as_list(), end='\n')
        
    return curr_layer

# Cost function
logits_train = model(keep_probs_train)
loss_train = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits_train, y))
train_prediction = tf.nn.softmax(logits_train)

logits = model(keep_probs_all)
valid_prediction = tf.nn.softmax(logits)  

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss_train)

#%% Prepare datasets
prop_test = 0.1
prop_validation = 0.1
n_data = n_pos + n_neg
n_test = np.int32(n_data * prop_test)
n_validation = np.int32(n_data * prop_validation)
n_train = n_data - n_test - n_validation

def reformat(dataset, labels):
    dataset = dataset.reshape(
        [-1] + [img_size] * 3 + [n_chan]).astype(np.float32)
    labels = (np.arange(n_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def get_data(n_pos, n_neg):
    # Load further data for negative examples.
    # Generate further data for positive examples.
    
    ix_data = np.arange(n_data)
    np.random.shuffle(ix_data)
    ix_test = ix_data[:n_test]
    ix_validation = ix_data[n_test:(n_test + n_validation)]
    ix_train = ix_data[(n_test + n_validation):]
                       
    return dataset, labels
    

    
def init():                   
#    #%% Start session
#    if 'sess' in vars():
#        sess.close()
        
    sess = tf.InteractiveSession()
    
    # Begin session
    sess.run(tf.initialize_all_variables())
    print('Initialized')
    step = 0
    
    return sess, step
    
#%%
    
    for step1 in range(n_steps):
        step += 1
      
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {x : batch_data, y : batch_labels}
        _, l, predictions = sess.run(
            [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
        if (step % validate_per_step == 0):
            print('Minibatch loss at step %d: %f' \
                  % (step, l))
            print('Minibatch accuracy: %.1f%%' \
                  % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' \
                  % accuracy(
                          valid_prediction.eval(
                              feed_dict = {x : valid_dataset}), 
                              valid_labels))
              
    return sess
              
#%%
print('Test accuracy: %.1f%%' % accuracy(
    valid_prediction.eval(
        feed_dict = {x : test_dataset}), test_labels))
