#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Gather all files into the same folder

Created on Mon Oct 17 22:43:53 2016

@author: yulkang
"""

import os
import shutil
import glob
import pandas as pd
import paths

#%%
dir_dst = os.path.join(paths.img_dir_root, 'all')
dir_subsets = glob.glob(os.path.join(paths.img_dir_root, 'subset*'))

if not os.path.isdir(dir_dst):
    os.mkdir(dir_dst)

#%%
df = pd.DataFrame(columns=['seriesuid', 'subset'])

for dir_subset in dir_subsets:
    files = [file1 \
             for file1 in os.listdir(dir_subset) \
             if os.path.isfile(os.path.join(dir_subset, file1))]
    for file_src in files:
        nam_ext = os.path.basename(file_src)
        uid, ext = os.path.splitext(nam_ext)
        
        full_src = os.path.join(dir_subset, file_src)
        full_dst = os.path.join(dir_dst, nam_ext)
        shutil.move(full_src, full_dst)
        print('Moved %s to %s' % (full_src, full_dst))
        
        if ext == '.mhd':
            df = df.append({'seriesuid':uid,
                            'subset':dir_subset[-1]}, 
                           ignore_index=True)
    os.rmdir(dir_subset)

file_csv = os.path.join(paths.img_dir_root, 'uid_subset.csv')
df.to_csv(file_csv, sep=',', index=False)
    