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
import pandas as pd
import warnings

from pysy.stat import ecdf
import compare_cand_vs_annot as annot

#%% Define functions
def load_itk_image(filename):
    if not os.path.isfile(filename):
        return None, None, None
    
    try:
        itkimage = sitk.ReadImage(filename)
        img_np = sitk.GetArrayFromImage(itkimage)
        origin_mm = np.array(list(reversed(itkimage.GetOrigin())))
        spacing_mm = np.array(list(reversed(itkimage.GetSpacing())))
        return img_np, origin_mm, spacing_mm
    except RuntimeError as err:
        warnings.warn('Error loading %s:' % filename)
        warnings.warn(err.message)
        return None, None, None
    
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
    
#%% Normalization
maxHU = 400.
minHU = -1000.

def normalizePlanes(npzarray):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray
    
#%% Paths
cand_file = 'Data/LUNA/candidates.csv'
annotation_file = 'Data/LUNA/annotations.csv'

#img_dir = 'Data/LUNA/image'
img_dir = '/Volumes/YKWD3GB/YulKang/LUNA/all'
#out_dir = 'Data/patches/'
out_dir = '/Volumes/YKWD3GB/YulKang/LUNA/patches'

subset_file = 'Data/LUNA/uid_subset.csv'
uid_file = 'Data/LUNA/uid.csv'
meta_file = 'Data/LUNA/img_meta.csv'

## an example uid that is in subset0
#uid0 = '1.3.6.1.4.1.14519.5.2.1.6279.6001.213140617640021803112060161074'

#%% load candidates
cands = annot.cands
cands_pos = annot.cands_pos

#%% Get unique uids
uid0 = cands.seriesuid
uids0 = cands.seriesuid.unique()

#%% uid2meta - example
def uid2mhd_file(uid):
    return os.path.join(img_dir, uid + '.mhd')

#%%
img_file = uid2mhd_file(uid0[0])
img_np, origin_mm, spacing_mm = load_itk_image(img_file)
print(img_np.shape)
print(origin_mm)
print(spacing_mm)

#%% Export metadata (shape, origin, spacing)
def uid2meta(uids=uids0):
    tbl = pd.DataFrame(columns=['uid',
                                'shape0', 'shape1', 'shape2',
                                'origin0', 'origin1', 'origin2',
                                'spacing0', 'spacing1', 'spacing2'])
    tbl.uid = uids
    print tbl.head()
    
    for i_row in range(len(tbl)):
        uid = uids[i_row]
        img_file = uid2mhd_file(uid)
        img_np, origin_mm, spacing_mm = load_itk_image(img_file)
        if img_np is None:
            continue
        else:
            row = list(img_np.shape) \
                + list(origin_mm) \
                + list(spacing_mm)
            print(row)
            tbl.iloc[i_row,1:] = np.array(row)
    
    tbl = tbl[~pd.isnull(tbl.shape0)]
    tbl.to_csv(meta_file, sep=',', index=False)
    return tbl
            
#%% Import uid2meta
if os.path.isfile(meta_file):
    print('meta_file exists. Loading uid2meta from %s' % meta_file)
    tbl = pd.read_csv(meta_file)
else:
    tbl = uid2meta(uids0)
          
#%% Exporting patches after interpolation
spacing_output_mm = np.array([1, 0.5, 0.5]) # ([5,5,5]) # 
size_output_mm = np.array([45, 45, 45]) # ([300,300,300]) # 
size_output_vox = size_output_mm / spacing_output_mm
delta_grid_mm = np.mgrid[
        -size_output_mm[0]/2:size_output_mm[0]/2:spacing_output_mm[0],
        -size_output_mm[1]/2:size_output_mm[1]/2:spacing_output_mm[1],
        -size_output_mm[2]/2:size_output_mm[2]/2:spacing_output_mm[2]]
delta_grid_mm = np.reshape(delta_grid_mm, (3,-1))

def uid2patch(uid):
    img_file = os.path.join(img_dir, uid + '.mhd')
    if not os.path.isfile(img_file):
        return 0
    
    img_np, origin_mm, spacing_mm = load_itk_image(img_file)
    print('Loaded ' + img_file)
    
    row_incl = np.nonzero(cands.seriesuid == uid)[0]
    
    for i_row in row_incl:
        cand2patch(cands.iloc[i_row], img_np, origin_mm, spacing_mm)
        break # DEBUG
    
    return 1

def cand2patch(cand, img_np=None, origin_mm=None, spacing_mm=None,
               is_annotation=True):
    from scipy.interpolate import interpn
    from pysy import zipPickle
    
    if img_np is None:
        uid = cand.seriesuid
        img_file = os.path.join(img_dir, uid + '.mhd')
        if not os.path.isfile(img_file):
            return 0
        
        img_np, origin_mm, spacing_mm = load_itk_image(img_file)
        print('Loaded ' + img_file)
    
    uid = cand.seriesuid
    cand_mm = np.reshape(
        np.asarray([cand.coordZ,
                    cand.coordY,
                    cand.coordX]),
        (3,1)) # z, y, x; bottom->up, ant->post, right->left
    
    if 'diamter' in cand.index:
        diameter_mm = float(cand.diamater)
    else:
        diameter_mm = -1
                         
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
                    fill_value=np.nan)
    patch = np.reshape(patch, size_output_vox)
    patch = normalizePlanes(patch)
    
    print origin_mm
    print patch.shape
    print np.min(patch)
    print np.max(patch)
    
    pth = os.path.join(
            out_dir, 
            'patch_' + uid
            + '_' + str(np.round(origin_mm[0])) 
            + '_' + str(np.round(origin_mm[1])) 
            + '_' + str(np.round(origin_mm[2])))
    
    # axial
    patch1 = patch[patch.shape[0]/2,:,:]
    plt.imshow(patch1, 
               cmap='gray')
    plt.show()
    
    # Save preview
    Image.fromarray(patch1*255).convert('L').save(pth + '.png')

    # coronal
    plt.imshow(patch[:,patch.shape[1]/2,:],
               cmap='gray')
    plt.show()
    
    # sagittal
    plt.imshow(patch[:,:,patch.shape[2]/2],
               cmap='gray')
    plt.show()
    
    zipPickle.save({'img':np.uint16(patch*(2**16-1)), 
                    'cand_mm':cand_mm, 
                    'origin_mm':origin_mm,
                    'spacing_mm':spacing_mm,
                    'diameter_mm':diameter_mm},
                   pth + '.zpkl')
    print('Saved to %s.zpkl' % pth)
    return 1

#%% Convert - demo
uid2patch(uid0[0])
    
#%% Convert - positive examples
n_to_convert = 3
for cand in cands_pos[1:]:
    if cand2patch(cand):
        n_to_convert -= 1
        if n_to_convert == 0:
            break

#%% Convert candidates to patches
for uid1 in uids0[:3]:
    uid2patch(uid1)
