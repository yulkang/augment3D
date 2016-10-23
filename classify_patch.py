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
from pysy import zipPickle

import compare_cand_vs_annot as annot
import import_mhd as mhd

#%% Choose train and test datasets
# Train: All positive in the subset + 3x negative (random subset)
cands_pos = annot.cands_pos
cands_neg = annot.cands_neg

subset_incl_pos = np.arange(9)
subset_incl_neg = np.array([0])

cands_pos = cands_pos.ix[cands_pos.subset.isin(subset_incl_pos),:]
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
    n_cand = len(cands)
    n_loaded = 0
    ix_loaded = np.zeros((n_cand), dtype=int32)
    
    for i_cand in range(n_cand):
        cand = cands.loc[i_cand,:]
        patch_file = mhd.cand_scale2patch_file(cand, scale, cand['is_pos'])
        if os.path.is_file(patch_file):
            L = zipPickle.load(patch_file)
            n_loaded += 1
        else:
            continue
        
        if n_loaded == 1:
            siz = np.concatenate(([n_cand], L.img.shape, [1]))
            img_all = np.zeros(siz, dtype=np.float32)
        
        img_all[n_loaded,:,:,:,0] = np.reshape(L['img'], )
        ix_loaded[n_loaded] = i_cand
            
    img_all = img_all[:n_loaded,:,:,:,:]
    ix_loaded = ix_loaded[:n_loaded]
    ix_loaded = cands.index(ix_loaded)
    label_all = cands.loc[ix_loaded,'is_pos']
        
    return img_all, label_all, ix_loaded

img_all, label_all, ix_loaded = load_cand(cands_all)
    
#%%
pickle_file = 'Data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
#%%
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#%%
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
#%%
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
  
#%%
num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  

#%% Problem 1
# Problem 1
# The convolutional model above uses convolutions with stride 2 
# to reduce the dimensionality. Replace the strides by a max pooling 
# operation (nn.max_pool()) of stride 2 and kernel size 2.

#%%
batch_size = 128 # 16
patch_size = 5
depth = 16
num_hidden = 64
  
sess = tf.InteractiveSession()

# Input data.
x = tf.placeholder(tf.float32, 
                   shape=[None, image_size, image_size, num_channels])
y = tf.placeholder(tf.float32, 
                   shape=[None, num_labels])

#tf_train_dataset = tf.placeholder(
#  tf.float32, shape=(batch_size, image_size, image_size, num_channels))
#tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#tf_valid_dataset = tf.constant(valid_dataset)
#tf_test_dataset = tf.constant(test_dataset)

# Variables.
keep_prob_train = [.5, 1, 1, 1]
keep_all = [1, 1, 1, 1]

layer_kind = ['conv', 'conv', 'dense', 'dense']
stride_conv = [1, 1, 0, 0]
width_conv = [5, 5, 0, 0]
width_pool = [2, 2, 0, 0]
depth_out = [16, 16, 16, 16]
depth_in = num_channels + depth_out[:-1]

weights0 = list()
biases = list()

n_layer = len(layer_kind)
for layer in range(n_layer):
  if layer_kind[layer] == 'conv':
    weights0.append(tf.Variable(tf.truncated_normal(
      [width_conv[layer], 
       width_conv[layer], 
       depth_in[layer], 
       depth_out[layer]],
      stddev = 0.1)))
    biases.append(tf.Variable(tf.constant(1.0, shape=[depth_out[layer]])))
  elif layer_kind[layer] == 'dense':
    
  else:
    raise(ValueError)
    

weights0.append(tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, num_channels, depth_conv[0]], stddev=0.1)))
biases.append(tf.Variable(tf.zeros([depth_conv[0]])))

weights0.append(tf.Variable(tf.truncated_normal(
    [patch_size, patch_size, depth_conv[0], depth_conv[1]], stddev=0.1)))
biases.append(tf.Variable(tf.constant(1.0, shape=[depth])))

weights0.append(tf.Variable(tf.truncated_normal(
    [image_size // stride_conv[0]**2 * image_size // stride_conv[0]**2 * depth, 
     num_hidden], stddev=0.1)))
biases.append(tf.Variable(tf.constant(1.0, shape=[num_hidden])))

weights0.append(tf.Variable(tf.truncated_normal(
    [num_hidden, num_labels], stddev=0.1)))
biases.append(tf.Variable(tf.constant(1.0, shape=[num_labels])))

# Model.
def add_conv2d(inp, weight, bias,
             keep_prob = 1, 
             stride = 2):
  
  weight_dropout = tf.nn.dropout(weight, keep_prob)
  stride_model = [1, stride, stride, 1]
  conv = tf.nn.conv2d(inp, weight_dropout, stride_model, 
                      padding='SAME')
  hidden = tf.nn.relu(conv + bias)
  hidden = tf.nn.max_pool(hidden, stride_model, stride_model, 
                          padding='SAME')
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
                        stride_conv[layer])
  
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


#%% Problem 2
# Try to get the best performance you can using a convolutional net. 
# Look for example at the classic LeNet5 architecture, 
# adding Dropout, and/or adding learning rate decay.

