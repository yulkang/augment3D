#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 21:46:48 2016

@author: yulkang
"""

#%% Import
import numpy as np

import pandas as pd
import warnings
import os
from PIL import Image
import matplotlib.pyplot as plt
import time

from pysy import zipPickle
import compare_cand_vs_annot as annot
import import_mhd as mhd

#%%
class Dataset(object):
    def __init__(self, cands,
                 is_pos = None,
                 scale = 0,
                 prop_test = 0.1,
                 prop_valid = 0.1,
                 **kwargs):
        self.cands = cands
        self.n_cands = len(self.cands)
        
        if is_pos is None:
            is_pos = self.cands.is_pos[0]
        
        self.prop_test = prop_test
        self.prop_valid = prop_valid
        self.n_test = np.ceil(self.n_cands * self.prop_test)
        self.n_train_valid = self.n_cands - self.n_test
        
        self.ix = np.arange(self.n_cands)
        np.random.shuffle(self.ix)
        self.ix_test = self.ix[:self.n_test]
        self.ix_train_valid = self.ix[self.n_test:]

        self.scale = scale
        self.output_format = mhd.output_formats.iloc[scale,:]
        _, self.img_size_in = mhd.output_format2size(
                self.output_format, self.is_pos)
        _, self.img_size_out = mhd.output_format2size(
                self.output_format, False)

        # Cache labels.
        self.labels = self.cands.loc[:,'is_pos']
        
        # Cache test images.
        self.__imgs_test = None
        
    def get_train_valid(self, n_samp):
        n_valid = np.ceil(n_samp * self.prop_valid)
        n_train = n_samp - n_valid
        
        imgs_valid, labels_valid = self.get_samples(n_valid)
        imgs_train, labels_train = self.get_samples(n_train)
        
        return imgs_train, labels_train, imgs_valid, labels_valid
        
    def get_test(self):
        if self.__imgs_test is None:
            # Load all
            imgs_test = self.load_imgs(self.cands.iloc[self.ix_test,:])
            self.__imgs_test = imgs_test
        else:
            imgs_test = self.__imgs_test
            
        labels_test = self.labels[self.ix_test]
        return imgs_test, labels_test

    def get_samples(self, n_samp):
        imgs = np.zeros([n_samp] + [self.img_size] * 3 + [1], 
                        dtype=np.float32)
        labels = np.zeros(n_samp, 
                          dtype=np.float32)
        n_retrieved = 0
        for n_retrieved in range(n_samp):
            img1, label1 = self.get_next_sample()
            
            imgs[n_retrieved - 1, :,:,:,:] = img1
            labels[n_retrieved - 1] = label1
            n_retrieved += 1
        
        return imgs, labels
        
    def get_next_sample(self):
        raise TypeError(
                'get_next_sample must be implemented in subclasses!')
        
    def load_imgs(self, cands, scale=0):    
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
#        patch1 = img_all[-1,img_all.shape[1]/2,:,:,0]
#        plt.imshow(patch1, cmap='gray')
#        plt.show()
        
        return img_all, label_all, ix_loaded
        
class DatasetNeg(Dataset):
    # Preload parts
    def __init__(self, cands, 
                 max_memory_MB = 2000, 
                 max_n_reuse = 1,
                 **kwargs):
        Dataset.__init__(self, cands, **kwargs)
        
        self.max_memory_MB = max_memory_MB
        self.max_n_reuse = max_n_reuse
        self.curr_n_reuse = 0
        
        self.n_used = np.zeros(self.n_train_valid, dtype=np.int32)
        self.is_loaded = np.zeros(self.n_train_valid, dtype=np.bool)

        self.ix_to_load_in_train_valid = None
        self.ix_to_read_in_train_valid = None
        
    def get_next_sample(self, scale):
        if self.ix_to_read_in_train_valid is None:
            self.load_next_samples(scale)
            
        
            
        ix_available = np.nonzero(self.is_loaded 
                                  & (self.n_used < self.curr_n_reuse))[0]
        if ix_available.size == 0:
            if self.curr_n_reuse < self.max_n_reuse:
                self.curr_n_reuse += 1
                return self.get_next_sample(scale)
            else:
                pass # Load more
        else:
            pass # Retrieve from cache
        
        img1, label1 = self.get_sample(
                self.ix_train_valid[self.curr_ix])
        self.curr_ix += 1
        if self.curr_ix == self.n_cands:
            np.random.shuffle(self.ix_train_valid)
            self.curr_ix = 0 # Loop around
        
        if img1 is not None:
            pass
        
        if self.is_loaded[self.ix_samp] and self.n_used[self.ix_samp]:
            self.imgs[self.ix_samp,:,:,:]
        else:
            pass
            
class DatasetPos(Dataset):
    # Load all on construction, augment by shift
    def __init(self, cands,
               **kwargs):
        pass
    
    def get_sample(self, ix_samp):
        # TODO
        pass

