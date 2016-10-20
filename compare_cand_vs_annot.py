#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 21:21:05 2016

@author: yulkang
"""

import pandas as pd
import numpy as np

cand_file = 'Data/LUNA/candidates.csv'
annotation_file = 'Data/LUNA/annotations.csv'
cand_out_file = 'Data/LUNA/candidates_vs_annot.csv'
uid_subset_file = 'Data/LUNA/uid_subset.csv'

#%% Load csvs
cands = pd.read_csv(cand_file)
cands_pos = pd.read_csv(annotation_file)
uid_subset = pd.read_csv(uid_subset_file)

#%% Compare cands and cands_pos
def main():
    changed_cands = False
    changed_cands = changed_cands or \
            match_cands(cands)

    changed_cands = changed_cands or \
            subset2cand(cands)
            
    if changed_cands:
        cands.to_csv(cand_out_file, sep=',', index=False)
        print('Saved to %s' % cand_out_file)
        
#%% 
def match_cands(cands1=cands):
    if np.all(pd.Series(['cand_ix','dist','radius']).isin(cands1.columns)):
        return 0
    
    for row in range(len(cands1)):
        cand_ix, dist, radius = match_cand(cands1.iloc[row])
        
        cands1.loc[row, 'cand_ix'] = cand_ix
        cands1.loc[row, 'dist'] = dist
        cands1.loc[row, 'radius'] = radius
        print(cands1.loc[row,:]) # DEBUG

        if cands1.loc[row,'class'] != (dist <= radius):
            print(cands1.loc[row,:])
            raise ValueError('Discrepancy of class was found')
    return 1

def match_cand(cand):
    same_uid = np.nonzero(cands_pos['seriesuid'] == cand['seriesuid'])[0]

    if same_uid.size == 0:
        cand_ix = np.nan
        dist = np.nan
        radius = np.nan
    else:
        cand_pos_same_uid = cands_pos.iloc[same_uid,:]
    
        d_coord = np.array(cand_pos_same_uid[['coordX', 'coordY', 'coordZ']]) \
                - np.array(cand[['coordX', 'coordY', 'coordZ']])
        dist = np.sum(d_coord ** 2, 1) ** 0.5
        radius = np.array(cand_pos_same_uid['diameter_mm']) / 2
        ix_min_dist = np.argmin(dist)
        
        dist = dist[ix_min_dist]
        cand_ix = same_uid[ix_min_dist]
        radius = radius[ix_min_dist]
    
    return cand_ix, dist, radius
    
#%%  
def subset2cand(cands1 = cands):
    if np.any(cands1.columns.isin(['subset'])):
        return 0
        
    for row in range(len(cands1)):
        same_uid = np.nonzero(cands1.loc[row,'seriesuid'] 
                              == uid_subset['seriesuid'])
        cands1.loc[row,'subset'] = uid_subset.loc[same_uid,'subset']
    return 1
    
