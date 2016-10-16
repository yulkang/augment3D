#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:47:21 2016

@author: yulkang
"""

#%%
import pydicom as dcm
from pysy.file import dirfiles, dirdirs

#%%
ds = dcm.read_file('../Data/LIDC/orig/img/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/000001.dcm')

dirs = dirdirs('../Data/LIDC/orig/img')