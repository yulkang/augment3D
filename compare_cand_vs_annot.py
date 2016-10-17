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

#%% Load cands and cands_pos file
cands = pd.read_csv(cand_file)
cands_pos = pd.read_csv(annotation_file)

#%% Find pos
ix = np.nonzero(cands['class'])

#%% 
def check_cand_w_annot(cand):
    same_uid = np.nonzero(cands_pos['seriesuid'] == cand['seriesuid'])[0]
    cand_pos_same_uid = cands_pos.iloc[same_uid,:]

    d_coord = np.array(cand_pos_same_uid[['coordX', 'coordY', 'coordZ']]) \
            - np.array(cand[['coordX', 'coordY', 'coordZ']])
    dist = np.sum(d_coord ** 2, 1) ** 0.5
    radius = np.array(cand_pos_same_uid['diameter_mm']) / 2
    is_pos = dist <= radius

    if cand['class'] != np.any(is_pos):
        print(cand)
        raise ValueError('Discrepancy of class was found')
    elif cand['class'] == 1:
        print('Found class==1 for uid: %s' % cand['seriesuid'])
        for is_pos1 in np.nonzero(is_pos)[0]:
            print('dist %1.1f <= radius %1.1f' % 
                  (dist[is_pos1], radius[is_pos1]))
            
#%% Compare cands and cands_pos
for row in range(len(cands)):
    check_cand_w_annot(cands.iloc[row])
        