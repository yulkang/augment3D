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

#%% Class
class DatasetPosNeg(object):
    #%% Initialization
    def __init__(self, cands_pos, cands_neg, 
               prop_pos = 0.5,
               pos_args = {},
               neg_args = {},
               **kwargs):
        # Give common arguments as kwargs. 
        # pos_args and neg_args overwrites kwargs.
        
        self.prop_pos = prop_pos
        
        pos_args1 = kwargs.copy()
        pos_args1.update(pos_args)
        
        neg_args1 = kwargs.copy()
        neg_args1.update(neg_args)

        self.ds_pos = DatasetPos(cands_pos, **pos_args1)
        self.ds_neg = DatasetNeg(cands_neg, **neg_args1)
        
    def get_train_valid(self, n_samp):
        # n_samp should be the size of a minibatch,
        # from which the gradient is calculated.
        # The order of positive and negative samples is not randomized!
        
        n_pos = np.int32(np.ceil(n_samp * self.prop_pos))
        n_neg = n_samp - n_pos
        
        imgs_train_pos, labels_train_pos, imgs_valid_pos, labels_valid_pos = \
                self.ds_pos.get_train_valid(n_pos)
                
        imgs_train_neg, labels_train_neg, imgs_valid_neg, labels_valid_neg = \
                self.ds_neg.get_train_valid(n_neg)
        
        return np.concatenate((imgs_train_pos, imgs_train_neg)), \
               np.concatenate((labels_train_pos, labels_train_neg)), \
               np.concatenate((imgs_valid_pos, imgs_valid_neg)), \
               np.concatenate((labels_valid_pos, labels_valid_neg))

class Dataset(object):
    #%% Initialization
    def __init__(self, cands,
                 is_pos = None,
                 scale = 0,
                 prop_test = 0.1,
                 prop_valid = 0.1,
                 **kwargs):
        self.cands = cands
        
        if is_pos is None:
            self.is_pos = self.cands.is_pos.iloc[0]
        else:
            self.is_pos = is_pos
        
        self.prop_test = prop_test
        self.prop_valid = prop_valid
        
        self.scale = scale
        self.output_format = mhd.output_formats.iloc[scale,:]
        _, self.img_size_in = mhd.output_format2size(
                self.output_format, self.is_pos)
        _, self.img_size_out = mhd.output_format2size(
                self.output_format, False)

        # Filter cands
        self.cands = self._filter_cands(self.cands)
        
        self.n_cands = len(self.cands)
        self.n_test = np.int32(np.ceil(self.n_cands * self.prop_test))
        self.n_train_valid = self.n_cands - self.n_test
        
        self.ix = np.arange(self.n_cands)
        np.random.shuffle(self.ix)
        self.ix_test = self.ix[:self.n_test]
        self.ix_train_valid = self.ix[self.n_test:]

        # Cache labels.
        self.labels = self.cands.loc[:,'is_pos']
        
        # Cache test images.
        self.imgs_test = None
        
    def _filter_cands(self, cands):
        return cands # Implement in subclasses
        
    #%% Retrieval - Interface
    def get_train_valid(self, n_samp):
        n_valid = np.int32(np.ceil(n_samp * self.prop_valid))
        n_train = n_samp - n_valid
        
#        print('n_valid: %d' % n_valid)
#        print('n_train: %d' % n_train)
        
        imgs_valid, labels_valid = self._get_samples(n_valid)
        imgs_train, labels_train = self._get_samples(n_train)
        
        return imgs_train, labels_train, imgs_valid, labels_valid
        
    def get_test(self):
        if self.imgs_test is None:
            # Load all
            imgs_test, _, _ = self._load_imgs(
                    self.cands.iloc[self.ix_test,:])
            self.imgs_test = imgs_test
        else:
            imgs_test = self.imgs_test
            
        labels_test = self.labels[self.ix_test]
        return imgs_test, labels_test

    #%% Retrieval - Internal
    def _get_samples(self, n_samp):
        
#        print('get_samples(%d)' % n_samp)
        
        imgs = np.zeros([n_samp] + [self.img_size_out] * 3 + [1], 
                        dtype=np.float32)
        labels = np.zeros([n_samp, 2], 
                          dtype=np.float32)
        n_retrieved = 0
        for n_retrieved in range(n_samp):
            img1, label1 = self._get_next_sample()
            
#            print('img1.shape:')
#            print(img1.shape)
#            print('label1.shape:')
#            print(label1.shape)
            
            img1 = img1 - np.mean(img1)
            
            std_img1 = np.std(img1)
            if std_img1 != 0:
                img1 = img1 / np.std(img1)

            imgs[n_retrieved - 1, :,:,:,:] = img1
            labels[n_retrieved - 1, 0] = label1
            n_retrieved += 1
        
        labels[:,1] = 1 - labels[:,0]
        return imgs, labels
        
    def _get_next_sample(self):
        raise TypeError(
                '_get_next_sample must be implemented in subclasses!')
        
    def _load_imgs(self, cands):    
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
                    output_format=self.output_format)
            if os.path.isfile(patch_file + '.zpkl'):
                L = zipPickle.load(patch_file + '.zpkl')
                n_loaded += 1
#                print('Loaded %d (%d/%d successful): %s.zpkl' % 
#                      (i_cand, n_loaded, n_cand, patch_file))
            else:
                print('Failed to find %s.zpkl' % patch_file)
                continue
            
            if n_loaded == 1:
                siz = np.concatenate(([n_cand], L['img'].shape))
                img_all = np.zeros(siz, dtype=np.float32)
                siz1 = siz.copy()
                siz1[0] = 1
#                print('Loading images of size:')
#                print(L['img'].shape)
                
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
            
#            print(img_all.shape)
#            print(siz)
            siz[0] = n_loaded
            
            img_all = img_all.reshape(np.concatenate((siz,[1])))
            ix_loaded = ix_loaded[:n_loaded]
            ix_loaded = cands.index[ix_loaded]
            label_all = cands.loc[ix_loaded,'is_pos']
        else:
            label_all = None
            ix_loaded = None
    
        t_el = time.time() - t_st
        if n_loaded == 0:
            t_el_per_loaded_ms = np.nan
        else:
            t_el_per_loaded_ms = t_el / n_loaded * 1e3
            
        print('Time elapsed: %1.2f sec / %d img = %1.2f msec/img' % 
              (t_el, n_loaded, t_el_per_loaded_ms))
        
        if n_loaded > 0:
            print('Last image loaded:')
            print(np.int32(ix_loaded[-1])) # DEBUG    
    #        patch1 = img_all[-1,img_all.shape[1]/2,:,:,0]
    #        plt.imshow(patch1, cmap='gray')
    #        plt.show()
        
        return img_all, label_all, ix_loaded
        
class DatasetNeg(Dataset):
    # Preload parts
    def __init__(self, cands, 
                 n_img_per_load = None,
                 max_memory_MB = 2000, 
                 max_n_reuse = 1,
                 **kwargs):
        
        Dataset.__init__(self, cands, **kwargs)
        
        self.max_memory_MB = max_memory_MB
        if n_img_per_load is None:
            self.n_img_per_load = np.round(
                    self.max_memory_MB \
                    / (self.img_size_in ** 3 * 4 * 1e3))
        else:
            self.n_img_per_load = n_img_per_load
        
        self.max_n_reuse = max_n_reuse
        self.curr_n_reuse = 0
        
        self.n_used_aft_load = 0
        self.ix_to_load = 0
        self.ix_to_read = -1
        
        self._load_next_samples()
        
    def _get_next_sample(self):
        self.ix_to_read += 1
        if self.ix_to_read == self.n_img_per_load:
            self.ix_to_read = 0
            self.n_used_aft_load += 1
            if self.n_used_aft_load > self.max_n_reuse:
                self._load_next_samples()
                
        return (self.imgs_train_valid[self.ix_to_read,:,:,:,:],
                self.labels_train_valid[self.ix_to_read])
        
    def _load_next_samples(self):
        n_loaded = 0
        img_all_size = [self.n_img_per_load] + [self.img_size_in] * 3 + [1]
        img_all = np.zeros(img_all_size, dtype=np.float32)
        labels = np.zeros(self.n_img_per_load, dtype=np.float32)
        while n_loaded < self.n_img_per_load:
            n_to_load = self.n_img_per_load - n_loaded
            ixs_to_load = self.ix_train_valid[
                    np.int32(
                            np.mod(self.ix_to_load + np.arange(n_to_load), 
                                   self.n_train_valid))]
            img_all1, labels1, _ = self._load_imgs(
                    self.cands.iloc[ixs_to_load,:])
            if img_all1 is None:
                n_loaded1 = 0
            else:
                n_loaded1 = img_all1.shape[0]
                
            if n_loaded1 > 0:
                self.ix_to_load = np.mod(self.ix_to_load + n_to_load,
                                         self.n_train_valid)
                ix_loaded = n_loaded + np.arange(n_loaded1)
                img_all[ix_loaded,:,:,:,:] = img_all1
                labels[ix_loaded] = labels1
                n_loaded += n_loaded1
#                print(img_all1.shape)
                
#            print('n_loaded/n_to_load: %d/%d' % (n_loaded, n_to_load))
        
        self.n_used_aft_load = 0
        self.imgs_train_valid = img_all
        self.labels_train_valid = labels
            
class DatasetPos(Dataset):
    # Load all on construction, augment by shift
    def __init__(self, cands,
               **kwargs):
        Dataset.__init__(self, cands, **kwargs)
        
        self.imgs_train_valid0, self.labels_train_valid, _ = \
                self._load_imgs(self.cands.iloc[self.ix_train_valid,:])
                
        self.radius = self.cands.radius
        self.spacing_output_mm = self.output_format.spacing_output_mm
        self.ix_vox = self.img_size_in / 2 \
              + np.arange(self.img_size_out) \
              - self.img_size_out / 2
                
        self.ix_to_read = -1
            
    def _filter_cands(self, cands):
        radius_min_incl = self.output_format.radius_min_incl
        radius_max_incl = self.output_format.radius_max_incl
        radius = np.float32(self.cands.radius)
        incl = (radius_min_incl <= radius) \
             & (radius < radius_max_incl)
        cands = cands.ix[incl,:]
        
        return cands
        
    def _get_next_sample(self):
        self.ix_to_read = np.mod(self.ix_to_read + 1, self.n_train_valid)
        ix1 = self.ix_to_read # shuffle happened in __init__
        
        radius_vox = self.radius.iloc[ix1] / self.spacing_output_mm
        dx, dy, dz = self._samp_sphere(radius_vox)
        x = np.int32(np.fix(dx)) + self.ix_vox
        y = np.int32(np.fix(dy)) + self.ix_vox
        z = np.int32(np.fix(dz)) + self.ix_vox
        
#        print('dx, dy, dz:')
#        print((dx, dy, dz))
#        print('x,y,z:')
#        print(x)
#        print(y)
#        print(z)
#        print('ix:')
#        print(ix1)
        ix_img = np.ix_(np.array([ix1]),x,y,z,np.array([0]))
#        print('ix_img:')
#        print(ix_img)
                    
        return (self.imgs_train_valid0[ix_img],
                self.labels.iloc[ix1])
    
    def _samp_sphere(self, radius = 1):
        # from http://stackoverflow.com/a/5408843/2565317
        
        if radius is not np.array:
            radius = np.array(radius)
            
        n = radius.size
        
        phi = np.random.rand(n) * 2 * np.pi
        costheta = np.random.rand(n) * 2 - 1
        u = np.random.rand(n)
        
        theta = np.arccos( costheta )
        r = radius * u ** (1. / 3)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return x, y, z
   
#%%
def demo_kwargs(args1, **kwargs):
    args1 = args1.copy()
    print(args1)
    args1.update(kwargs)
    print(args1)
        
    
#%%
def get_dataset():
    import datasets
    reload(datasets)
    cands_neg = mhd.cands_neg
    cands_neg = cands_neg.ix[cands_neg.subset.isin([0])]
    return datasets.DatasetPosNeg(mhd.cands_pos, cands_neg,
                              n_img_per_load = 1000)
    
#%%
def demo():
    #%% Test ds_neg
    import datasets
    reload(datasets)
    cands_neg = mhd.cands_neg
    cands_neg = cands_neg.ix[cands_neg.subset.isin([0])]
    
    #%%
    ds_neg = datasets.DatasetNeg(cands_neg, n_img_per_load = 50)
    
    #%%
    imgs_train, labels_train, imgs_valid, labels_valid = \
            ds_neg.get_train_valid(10)

    #%% Test ds_pos
    import datasets
    reload(datasets)
    ds_pos = datasets.DatasetPos(mhd.cands_pos[:100], n_img_per_load = 50)
    
    imgs_train, labels_train, imgs_valid, labels_valid = \
            ds_pos.get_train_valid(10)
    
    #%% Test ds_pos_neg
    import datasets
    reload(datasets)
    cands_neg = mhd.cands_neg
    cands_neg = cands_neg.ix[cands_neg.subset.isin([0])]
    ds_all = datasets.DatasetPosNeg(mhd.cands_pos[:100], cands_neg,
                              n_img_per_load = 50)
    
    imgs_train, labels_train, imgs_valid, labels_valid = \
            ds_all.get_train_valid(20)
            
#%%
if __name__ == '__main__':
    demo()