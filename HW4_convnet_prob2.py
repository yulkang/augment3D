#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:09:43 2016

@author: yulkang
"""

#%%
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

#%%
pickle_file = 'notMNIST.pickle'

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
sess.close()
  
#%% Problem 2
# Try to get the best performance you can using a convolutional net. 
# Look for example at the classic LeNet5 architecture, 
# adding Dropout, and/or adding learning rate decay.

sess = tf.InteractiveSession()

#%%
batch_size = 128 # 16
patch_size = 5
depth = 16
num_hidden = 64
  
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

#%% Variables.
keep_prob_train = [.8, .8, .8, .8, .8]
keep_all = [1, 1, 1, 1, 1]

layer_kind = ['conv', 'conv', 'conv', 'dense', 'dense']
stride_conv = [2, 2, 2, 1, 1]
width_conv = [5, 5, 5, 1, 1]
width_pool = [2, 2, 2, 1, 1]
depth_out = [8, 16, 32, 64, num_labels]
depth_in = [num_channels] + depth_out[:-1]

shape_in = [image_size, image_size, num_channels]

weights0 = list()
biases = list()

n_layer = len(layer_kind)
for layer in range(n_layer):
  print('layer:')
  print(layer)
  print(layer_kind[layer])
  print('shape_in:')
  print(shape_in)
  
  if layer_kind[layer] == 'conv':
    shape_weights = [
      width_conv[layer], 
      width_conv[layer], 
      depth_in[layer], 
      depth_out[layer]]
    weights0.append(tf.Variable(tf.truncated_normal(
      shape_weights,                                                        
      stddev = 0.1)))
    biases.append(tf.Variable(tf.constant(1.0, shape=[depth_out[layer]])))
    
    shape_in1 = np.int32(np.round(np.double(shape_in[0]) 
                                / np.double(stride_conv[layer])))
    shape_in = [shape_in1, shape_in1, depth_out[layer]]
    
  elif layer_kind[layer] == 'dense':
    shape_weights = [np.prod(shape_in), depth_out[layer]]
    weights0.append(tf.Variable(tf.truncated_normal(
      shape_weights,
      stddev = 0.1)))
    biases.append(tf.Variable(tf.constant(1.0, shape=[depth_out[layer]])))
    shape_in = [depth_out[layer]]
          
  else:
    raise(ValueError)
    
  print('shape_weights: ')
  print(shape_weights)
    
#%% Model.
def add_conv2d(inp, weight, bias,
             keep_prob = 1, 
             stride_conv = 2,
             width_pool = 2):
  
  weight_dropout = tf.nn.dropout(weight, keep_prob)
  stride_model = [1, stride_conv, stride_conv, 1]

#  print(inp)
#  print(weight_dropout)

  conv = tf.nn.conv2d(inp, weight_dropout, stride_model, 
                      padding='SAME')
  hidden = tf.nn.relu(conv + bias)
  hidden = tf.nn.max_pool(hidden, 
                          [1, width_pool, width_pool, 1], 
                          [1, 1, 1, 1], 
                          padding='SAME')
  return hidden
  
def add_dense(inp, weight, bias, keep_prob = 1, is_final = False):
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
  n_layer = len(layer_kind)
  inp = x
  
  for layer in range(n_layer):
    print('layer %d' % layer)
    print('inp:')
    print(inp)
    print('weights:')
    print(weights[layer])
    print('-----')
    
    if layer_kind[layer] == 'conv':
      inp = add_conv2d(inp, weights[layer], biases[layer], keep_prob[layer], 
                          stride_conv[layer], width_pool[layer])
      
    elif layer_kind[layer] == 'dense':
      inp = add_dense(inp, weights[layer], biases[layer], keep_prob[layer],
                         is_final = layer == n_layer)
  
  return inp

# Training computation / Optimizer
print('logits_train')
logits_train = model(keep_prob_train)
loss_train = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(logits_train, y))
train_prediction = tf.nn.softmax(logits_train)

print('logits_valid')
logits = model(keep_all)
valid_prediction = tf.nn.softmax(logits)  

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss_train)

#%%
sess.run(tf.initialize_all_variables())
print('Initialized')
step = 0

#%%
max_num_steps = 30001
num_steps = 2001 # 3000 steps give test accuracy 91.5%
display_per_step = 100

accu_valid_prev = 0
accu_valid = 0

#%%
while (step < max_num_steps) \
    and ((step == 0) or (accu_valid > accu_valid_prev)):
    
  accu_valid_prev = accu_valid
  
  for step1 in range(num_steps):
    step += 1
    
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {x : batch_data, y : batch_labels}
    _, l, predictions = sess.run(
      [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
    if (step % display_per_step == 0):
      accu_valid = accuracy(
        valid_prediction.eval(
          feed_dict = {x : valid_dataset}), 
          valid_labels)
      
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accu_valid)
  
#%%
#for step1 in range(num_steps):
#  step += 1
#  
#  offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#  batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#  batch_labels = train_labels[offset:(offset + batch_size), :]
#  feed_dict = {x : batch_data, y : batch_labels}
#  _, l, predictions = sess.run(
#    [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
#  if (step % display_per_step == 0):
#    accu_valid = accuracy(
#      valid_prediction.eval(
#        feed_dict = {x : valid_dataset}), 
#        valid_labels)
#    
#    print('Minibatch loss at step %d: %f' % (step, l))
#    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#    print('Validation accuracy: %.1f%%' % accu_valid)
    
#%%
print('Test accuracy: %.1f%%' % accuracy(
  valid_prediction.eval(
    feed_dict = {x : test_dataset}), test_labels))

#%%
w = tf.Variable(tf.truncated_normal([1,3,2,4], stddev=0.1))

#%%
# sess.close()
