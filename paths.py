#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 04:32:01 2016

@author: yulkang
"""

# Input - small
cand_file = '../Data/LUNA/candidates.csv'
annotation_file = '../Data/LUNA/annotations.csv'
uid_subset_file = '../Data/LUNA/uid_subset.csv'

# Input - large
img_dir = '/Volumes/YK_SSD_1TB/LUNA/all'

# Output - small
subset_file = '../Data/LUNA/uid_subset.csv'
meta_file = '../Data/LUNA/img_meta.csv'
cand_out_file = '../Data/LUNA/candidates_vs_annot.csv'

# Output - large
patch_dir = '/Volumes/YK_SSD_1TB/LUNA/patches'
