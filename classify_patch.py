#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains a deep 3D convolution network to classify 3D patches.
Use as a demo for how to retrieve data for training.
Hyperparameters are yet to be optimized.

Some parts are from TensorFlow Udacity Tutorial, available at:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb

Created on Sat Jun 11 17:09:43 2016

@author: yulkang
"""

#%% Import
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range

import datasets

#%% Import dataset
# Requires LUNA data. See the Installation section of readme.md.
reload(datasets)
ds = datasets.get_dataset(prop_valid = 0)

#%% Functions to build the network
def accuracy(predictions, labels):
    return (100.0 * np.mean((predictions[:,1] > predictions[:,0]) == labels[:,1]))
  
def add_conv(inp, 
             width_mult, 
             stride_mult,
             depth_out,
             width_pool = 0,
             stride_pool = 2,
             stddev = 0.1,
             bias = 0.0,
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
    return hidden, weight_dropout
  
def add_dense(inp, depth_out, 
              bias = 0.0,
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
    return dense, weight_dropout

def main():
    #%% Hyperparameters. Yet to be optimized.
    img_size = ds.ds_pos.img_size_out
    
    n_labels = 2
    n_chan = 1 # grayscale
    batch_size = 16
    patch_size = 3
    stride_size = 2
    pool_size = 2
    depth_conv = 64
    n_hidden = 64
      
    # Input data.
    x = tf.placeholder(tf.float32, 
                       shape=[None, img_size, img_size, img_size, n_chan])
    y = tf.placeholder(tf.float32, 
                       shape=[None, n_labels])
    
    # Variables.
    keep_probs_train = [1, 1, .9, .9] # .5, .5]
    keep_probs_all = np.ones(len(keep_probs_train))
    loss_reg_wt = [0.001, 0.001, 0.001, 0.001]
    
    layer_kind = ['conv', 'conv', 'dense', 'dense']
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
        """ 
        Gives a model for either training or validation depending on keep_probs
        which determines whether to apply dropout 
        (give 1s, or keep_probs_all, for validation).
        """
        
        all_net = x
        loss_reg = tf.Variable(0.0)
        for layer in range(n_layer):
            print('--- Layer %d ---' % layer)
            print('Input :', end='')
            print(all_net.get_shape().as_list(), end='\n')
            
            if layer_kind[layer] == 'conv':
                all_net, curr_wt = add_conv(
                        all_net,
                        widths_mult[layer],
                        strides_mult[layer],
                        depth_out[layer],
                        width_pool=widths_pool[layer],
                        stride_pool=strides_pool[layer],
                        keep_prob=keep_probs[layer])
                loss_reg = loss_reg + tf.nn.l2_loss(curr_wt) * loss_reg_wt[layer]
                
            elif layer_kind[layer] == 'dense':
                all_net, curr_wt = add_dense(
                        all_net,
                        depth_out[layer],
                        keep_prob=keep_probs[layer],
                        is_final=layer >= n_layer - 1)
                loss_reg = loss_reg + tf.nn.l2_loss(curr_wt) * loss_reg_wt[layer]
            else:
                raise ValueError('layer_kind[%d]=%s not allowed!' % 
                                 (layer, layer_kind[layer]))
            
            print('Output:', end='')
            print(all_net.get_shape().as_list(), end='\n')
            
        return all_net, loss_reg
    
    # Cost function
    logits_train, loss_reg_train = model(keep_probs_train)
    loss_train = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits_train, y)
        + loss_reg_train)
    train_prediction = tf.nn.softmax(logits_train)
    
    logits, _ = model(keep_probs_all)
    valid_prediction = tf.nn.softmax(logits)  
    
    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss_train)
    
    #%% Start a session
    if 'sess' in vars():
        sess.close() # For easy experimentation using the cell mode in Spyder.
            
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    print('Initialized')
    step = 0
        
    #%%
    max_num_steps = 50
    num_steps = 10
    validate_per_step = 1
    
    accu_valid_prev = -1
    accu_valid = 0
    
    batch_size = 100
    
    def colvec(v):
        return np.reshape(v, [-1,1])
    
    #%% Test run
    imgs_train, labels_train, imgs_valid, labels_valid = \
        ds.get_train_valid(batch_size)
    
    feed_dict = {x: imgs_train, y: labels_train}
    _, l, predictions = sess.run(
        [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
    
    logits = sess.run(logits_train, feed_dict=feed_dict)
    
    print('loss: %f' % l)
    print('mean predictions: %f' % np.mean(predictions))
    print('logits:')
    print(logits)
    
    #%% Iterate over minibatches
    while (step < max_num_steps) \
        and ((step == 0) or (accu_valid > accu_valid_prev)):
        
        accu_valid_prev = accu_valid
        
        for step1 in range(num_steps):
            step += 1
            
            imgs_train, labels_train, _, _ = \
                ds.get_train_valid(batch_size)
            
            feed_dict = {x: imgs_train, y: labels_train}
            _, l, predictions = sess.run(
                [optimizer, loss_train, train_prediction], feed_dict=feed_dict)
            
            print('Minibatch loss at step %d: %f' \
                  % (step, l))
            print('Minibatch accuracy: %.1f%%' \
                  % accuracy(predictions, labels_train))
            
            if (step % validate_per_step == 0):
                imgs_valid, labels_valid, _, _ = \
                    ds.get_train_valid(batch_size)
                    
                pred_valid = valid_prediction.eval(
                                  feed_dict = {x : imgs_valid})
                accu_valid = accuracy(pred_valid, labels_valid)
                print('Validation accuracy: %.1f%%' \
                      % accu_valid)
                  
    #%% Test accuracy
    imgs_test, labels_test = ds.get_test()
    print('Test accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(
            feed_dict = {x : imgs_test}), labels_test))

#%%
if __name__ == '__main__':
    main()