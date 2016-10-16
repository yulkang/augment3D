#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
demo_export_tiff_multipage

Created on Sun Oct 16 12:11:01 2016

@author: yulkang
"""

#%%
import numpy as np
from PIL import Image
import os
import tifffile as tif

patch = np.random.rand(13,65,65)
img_dir = os.path.join('Data', 'demo_export_tiff_multipage')
img_name = '%d_%d_%d' % (patch.shape[0],
                         patch.shape[1],
                         patch.shape[2])

#%%
# Image.fromarray(patch*255).convert('L').save(pth)
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)

tif.imsave(img_file, patch)

#%%
tif.imshow(patch[1,:,:])

#%%
patch1 = tif.imread(img_file)

#%%
img_tif = tif.TiffFile(img_file)

#%% Use zipPickle instead. Smaller than tif and preserves metadata.
from pysy import zipPickle

#%%
img_pkl = os.path.join(img_dir, img_name + '.zpkl')

zipPickle.save({'img':patch}, img_pkl)

#%%
L = zipPickle.load(img_pkl)

