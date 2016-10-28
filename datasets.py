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
                 prop_test = 0.1,
                 prop_valid = 0.1,
                 max_memory_MB = 2000):
        self.cands = cands
        
        self.n_test = np.ceil(self.n_cands * prop_test)
        self.prop_valid = prop_valid
        
        self.ix = np.arange(self.n_cands)
        np.random.shuffle(self.ix)
        self.ix_test = self.ix[:self.n_test]
        self.ix_train_valid = self.ix[self.n_test:]
        self.curr_ix = 0
        self.is_loaded = np.zeros(np.n_cands, dtype=np.bool)

        self.imgs = None
        self.labels = self.cands.loc[:,'is_pos']
        self.max_memory_MB = max_memory_MB
        
    @property
    def n_cands(self):
        return len(self.cands)
        
    def get_train_valid(self, n_samp):
        n_valid = np.ceil(n_samp * self.prop_valid)
        n_train = n_samp - n_valid
        
        imgs_valid, labels_valid = self.get_samples(n_valid)
        imgs_train, labels_train = self.get_samples(n_train)
        
        return imgs_train, labels_train, imgs_valid, labels_valid
        
    def get_samples(self, n_samp):
        imgs = np.zeros([n_samp] + [self.img_size] * 3 + [1], 
                        dtype=np.float32)
        labels = np.zeros(n_samp, 
                          dtype=np.float32)
        n_retrieved = 0
        while n_retrieved < n_samp:
            img1, label1 = self.get_sample(
                    self.ix_train_valid[self.curr_ix])
            self.curr_ix += 1
            if self.curr_ix == self.n_cands:
                np.random.shuffle(self.ix_train_valid)
                self.curr_ix = 0 # Loop around
            
            if img1 is not None:
                imgs[n_retrieved - 1, :,:,:,:] = img1
                labels[n_retrieved - 1] = label1
                n_retrieved += 1
        
        return imgs, labels
        
    def get_sample(self, ix_samp):
        raise TypeError('get_sample must be modified in subclasses!')

class DatasetNeg(Dataset):
    # Load by subset?
    def get_sample(self, ix_samp):
        # TODO
        if self.is_loaded[ix_samp]:            
            self.imgs[ix_samp,:,:,:]
        else:
            pass # Load from disk
        pass
            
class DatasetPos(Dataset):
    # Augment by shift
    def get_sample(self, ix_samp):
        # TODO
        pass

