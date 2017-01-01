#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Checks if positive candidates indeed have the 

Created on Sun Oct 16 21:21:05 2016

@author: yulkang
"""

import pandas as pd
import numpy as np
import os
import paths

#%% Load csvs
cands = pd.read_csv(paths.cand_file)
cands_pos = pd.read_csv(paths.annotation_file)
n_pos = len(cands_pos)
cands_pos.loc[:,'is_pos'] = np.ones(n_pos, dtype=np.bool)
cands_pos.loc[:,'dist'] = np.zeros(n_pos, dtype=np.float16)
cands_pos.loc[:,'ix_pos'] = cands_pos.index
cands_pos.loc[:,'radius'] = cands_pos.loc[:,'diameter_mm']/2

uid_subset = pd.read_csv(paths.uid_subset_file)

uids = cands_pos.seriesuid.unique()

if os.path.isfile(paths.cand_out_file):
    cands_vs_annot = pd.read_csv(paths.cand_out_file)
else:
    print('Run compare_cand_vs_annot.main() to save cands_vs_annot at %s' \
          % paths.cand_out_file)

cands_vs_annot.loc[:,'is_pos'] = cands_vs_annot.loc[:,'is_pos']==1
cands_neg = cands_vs_annot.ix[~cands_vs_annot.is_pos,:]

n_cand = len(cands)
n_cand_pos = len(cands_pos)

coords = np.transpose(np.array(cands.loc[:,['coordX','coordY','coordZ']]))
coords_pos = np.transpose(np.array(cands_pos.loc[:,['coordX','coordY','coordZ']]))

#%% Compare cands and cands_pos
def main():
    changed_cands = False
    changed_cands = changed_cands or match_uids(uids)
    changed_cands = changed_cands or subset2cand(cands)
            
    if changed_cands:
        cands.to_csv(paths.cand_out_file, sep=',', index=False)
        print('Saved to %s' % paths.cand_out_file)
        
#%% 
def match_uids(uids1):
#    if np.all(pd.Series(['dist',
#                         'radius',
#                         'is_pos',
#                         'ix_pos'
#                         ]).isin(cands.columns)):
#        return 0
        
    dist = np.nan + np.zeros(n_cand)
    radius = np.nan + np.zeros(n_cand)
    is_pos = np.zeros(n_cand)
    ix_pos = np.nan + np.zeros(n_cand)
        
    for uid1 in uids1:
        in_cand = np.nonzero(cands.seriesuid == uid1)[0]
        in_cand_pos = np.nonzero(cands_pos.seriesuid == uid1)[0]
        
        if not np.any(in_cand_pos):
            return 0
            
#        print(uid1)
#        print(len(in_cand_pos))
#        print(coords_pos.shape)
#        print(coords_pos[:,in_cand_pos].shape)
        
        d_coords = np.reshape(coords[:,in_cand], [3,-1,1]) \
                 - np.reshape(coords_pos[:,in_cand_pos], [3,1,-1],
                              order='F')
                 
#        print(d_coords.shape)
                 
        dist1 = np.sum(d_coords ** 2, 0) ** 0.5
#        print(dist1.shape)
        
        radius1 = np.reshape(cands_pos.loc[in_cand_pos,'diameter_mm'] / 2, 
                             [1,-1])
        is_pos1 = dist1 <= radius1
        
        min_ix = np.argmin(dist1, 1)
        min_dist = np.min(dist1, 1)
        
        dist[in_cand] = min_dist
        radius[in_cand] = radius1
        is_pos[in_cand] = np.any(is_pos1, 1)
        ix_pos[in_cand] = in_cand_pos[min_ix]
        
    cands.loc[:,'dist'] = dist
    cands.loc[:,'radius'] = radius
    cands.loc[:,'is_pos'] = is_pos
    cands.loc[:,'ix_pos'] = ix_pos
        
    return np.sum(is_pos)
    
#%%  
def subset2cand(cands1 = cands):
    if np.any(cands1.columns.isin(['subset'])):
        return 0
                
    subsets1 = np.zeros(len(cands1))
        
    for subset1 in uid_subset.subset.unique():
        in_subset1 = uid_subset.subset == subset1
        uid_subset1 = uid_subset.ix[in_subset1,'seriesuid']
        same_uid = np.nonzero(cands1.seriesuid.isin(uid_subset1))
        subsets1[same_uid] = subset1

    cands1.loc[:,'subset'] = subsets1
    return 1
