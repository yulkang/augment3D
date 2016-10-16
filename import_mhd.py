#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Mostly follows SimpleITKTutorial.ipynb
but loads corresponding file for each cand,
and fill with 0 if the nodule is too close to the boundary.

Created on Sat Oct 15 16:22:52 2016

@author: yulkang
"""

#%% Import
import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
import pandas as pd

#%% Define functions
def load_itk_image(filename):
    if not os.path.isfile(filename):
        return None, None, None
    
    itkimage = sitk.ReadImage(filename)
    img_np = sitk.GetArrayFromImage(itkimage)
    origin_mm = np.array(list(reversed(itkimage.GetOrigin())))
    spacing_mm = np.array(list(reversed(itkimage.GetSpacing())))
    return img_np, origin_mm, spacing_mm
    
def readCSV(filename):
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines
    
#def worldToVoxelCoord(worldCoord, origin, spacing):
#    stretchedVoxelCoord = np.absolute(worldCoord - origin)
#    voxelCoord = stretchedVoxelCoord / spacing
#    return voxelCoord
    
def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray
    
#%% Set path
cand_file = 'Data/LUNA/candidates.csv'
img_dir = 'Data/LUNA/image'
uid_file = 'Data/LUNA/uid.csv'
meta_file='Data/LUNA/img_meta.csv'

# an example uid that is in subset0
uid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.213140617640021803112060161074'
#uid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.231645134739451754302647733304'

#%% load candidates
cands = readCSV(cand_file)

#%% Get unique uids
uid0 = list()
for cand in cands[1:]:
    uid0.append(cand[0])
uids0 = list(set(uid0))

#%% uid2meta - example
def uid2mhd_file(uid):
    return os.path.join(img_dir, uid + '.mhd')

img_file = uid2mhd_file(uid)
img_np, origin_mm, spacing_mm = load_itk_image(img_file)
print(img_np.shape)
print(origin_mm)
print(spacing_mm)

#%% Export metadata (shape, origin, spacing)
def uid2meta(uids=uids0):
    with open(meta_file, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['uid', 
                            'shape0', 'shape1', 'shape2',
                            'origin0', 'origin1', 'origin2',
                            'spacing0', 'spacing1', 'spacing2'])
        
        for uid in uids:
            img_file = uid2mhd_file(uid)
            img_np, origin_mm, spacing_mm = load_itk_image(img_file)
            if img_np is None:
                continue
            else:
                row = [uid] + list(img_np.shape) \
                            + list(origin_mm) \
                            + list(spacing_mm)
                print(row)
                csvwriter.writerow(row)
            
#%% Export uid2meta
uid2meta(uids0)
             
#%% Functions to export patches after interpolation
spacing_output_mm = np.array([2.5, 0.5, 0.5])
size_output_mm = np.array([32.5, 32.5, 32.5])
size_output_vox = size_output_mm / spacing_output_mm
delta_grid_mm = np.mgrid[-size_output_vox[0]/2:size_output_vox[0]/2,
                 -size_output_vox[1]/2:size_output_vox[1]/2,
                 -size_output_vox[2]/2:size_output_vox[2]/2]
delta_grid_mm = np.reshape(delta_grid_mm, (3,-1))

#%%
def uid2patch(uid):
    img_file = os.path.join(img_dir, uid + '.mhd')
    if not os.path.isfile(img_file):
        return 0
    
    img_np, origin_mm, spacing_mm = load_itk_image(img_file)
    print('Loaded ' + img_file)
    
    for cand in cands[1:]:
        if cand[0] == uid:
            cand2patch(cand, img_np, origin_mm, spacing_mm)
            break # DEBUG
    
    return 1

def cand2patch(cand, img_np, origin_mm, spacing_mm):
    from scipy.interpolate import interpn
    from pysy import zipPickle
    
    uid = cand[0]
    cand_mm = np.reshape(
        np.asarray([float(cand[3]),float(cand[2]),float(cand[1])]),
        (3,1))
    
    grid0_mm = range(3)
    for dim in range(3):
        grid0_mm[dim] = np.arange(img_np.shape[dim]) \
                * spacing_mm[dim] \
                + origin_mm[dim]

    grid_mm = np.transpose(delta_grid_mm + cand_mm)
    
    print(cand)
#    print(grid0_mm)
#    print(grid_mm.shape)
#    print(grid_mm)
    
    patch = interpn(grid0_mm, img_np, grid_mm, 
                    bounds_error=False,
                    fill_value=0)
    patch = np.reshape(patch, size_output_vox)
    patch = normalizePlanes(patch)
    
    print origin_mm
    print patch.shape
    print np.min(patch)
    print np.max(patch)
    
    outputDir = 'Data/patches/'
    mid_slice_vox = patch.shape[0]/2;
    
    plt.imshow(patch[0,:,:], cmap='gray')
    plt.show()
    
    plt.imshow(patch[mid_slice_vox-1,:,:], cmap='gray')
    plt.show()
    
    plt.imshow(patch[mid_slice_vox,:,:], cmap='gray')
    plt.show()
    
    plt.imshow(patch[mid_slice_vox+1,:,:], cmap='gray')
    plt.show()
    
    plt.imshow(patch[-1,:,:], cmap='gray')
    plt.show()
    
    pth = os.path.join(
            outputDir, 
            'patch_' + uid
            + '_' + str(origin_mm[0]) 
            + '_' + str(origin_mm[1]) 
            + '_' + str(origin_mm[2])
            + '.zpkl')
    
    zipPickle.save({'img':patch*255, 
                    'cand_mm':cand_mm, 
                    'origin_mm':origin_mm,
                    'spacing_mm':spacing_mm},
                   pth)
    print('Saved to %s' % pth)

#%% Convert - demo
uid2patch(uid)
    
#%% Convert candidates to patches
for uid in uids:
    uid2patch(uid)
