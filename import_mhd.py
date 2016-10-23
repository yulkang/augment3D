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
cand_out_file = 'Data/LUNA/candidates_vs_annot.csv'
uid_subset_file = 'Data/LUNA/uid_subset.csv'

#img_dir = 'Data/LUNA/image'
img_dir = '/Volumes/YK_SSD_1TB/YulKang/LUNA/all'
#out_dir = 'Data/patches/'
out_dir = '/Volumes/YK_SSD_1TB/YulKang/LUNA/patches'

subset_file = 'Data/LUNA/uid_subset.csv'
meta_file = 'Data/LUNA/img_meta.csv'

## an example uid that is in subset0
#uid0 = '1.3.6.1.4.1.14519.5.2.1.6279.6001.213140617640021803112060161074'

#%% load candidates
cands = annot.cands
cands_pos = annot.cands_pos
uid_subset = annot.uid_subset
cands_vs_annot = annot.cands_vs_annot
cands_neg = cands.loc[cands.loc[:,'class'] != 1, :]

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
# max radius is 16.14mm
radius_range = [[0, 4], [4, 8], [8, 17]]

spacing_output_mm = np.array([0.5, 0.5, 0.5]) # ([5,5,5]) # 
size_output_mm = np.array([60, 60, 60]) # ([300,300,300]) # 
size_output_vox = size_output_mm / spacing_output_mm
delta_grid_mm = np.mgrid[
        -size_output_mm[0]/2:size_output_mm[0]/2:spacing_output_mm[0],
        -size_output_mm[1]/2:size_output_mm[1]/2:spacing_output_mm[1],
        -size_output_mm[2]/2:size_output_mm[2]/2:spacing_output_mm[2]]
delta_grid_mm = np.reshape(delta_grid_mm, (3,-1))

def uid2patch(uid, cands1):
    img_file = os.path.join(img_dir, uid + '.mhd')
    if not os.path.isfile(img_file):
        return 0
    
    img_np, origin_mm, spacing_mm = load_itk_image(img_file)
    print('Loaded ' + img_file)
    
    row_incl = np.nonzero(cands1.seriesuid == uid)[0]
    
    for i_row in row_incl:
        cand2patch(cands1.iloc[i_row], img_np, origin_mm, spacing_mm)
        break # DEBUG
    
    return 1

def cand2patch(cand, img_np=None, origin_mm=None, spacing_mm=None,
               is_annotation=True):
    from scipy.interpolate import interpn
    from pysy import zipPickle
    
    uid = cand.seriesuid
    cand_mm = np.reshape(
        np.asarray([cand.coordZ,
                    cand.coordY,
                    cand.coordX]),
        (3,1)) # z, y, x; bottom->up, ant->post, right->left
    
    pth = os.path.join(
            out_dir, 
            'patch_' + uid
            + '_' + str(np.round(origin_mm[0])) 
            + '_' + str(np.round(origin_mm[1])) 
            + '_' + str(np.round(origin_mm[2])))
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    if os.path.isfile(pth + '.zpkl') and \
            os.path.isfile(pth + '_slice0.png') and \
            os.path.isfile(pth + '_slice1.png') and \
            os.path.isfile(pth + '_slice2.png'):
        print('Exists already. Skipping %s' % pth)
        return 0        
    
    if img_np is None:
        uid = cand.seriesuid
        img_file = os.path.join(img_dir, uid + '.mhd')
        if not os.path.isfile(img_file):
            return 0
        
        img_np, origin_mm, spacing_mm = load_itk_image(img_file)
        print('Loaded ' + img_file)
    
    grid0_mm = range(3)
    for dim in range(3):
        grid0_mm[dim] = np.arange(img_np.shape[dim]) \
                * spacing_mm[dim] \
                + origin_mm[dim]

    grid_mm = np.transpose(delta_grid_mm + cand_mm)
    
    if 'diamter' in cand.index:
        diameter_mm = float(cand.diamater)
    else:
        diameter_mm = -1
                         
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
    
    # axial
    patch1 = patch[patch.shape[0]/2,:,:]
    plt.imshow(patch1, 
               cmap='gray')
    plt.show()
    
    # Save preview
    fout = pth + '_slice0.png'
    if not os.path.isfile(fout):
        Image.fromarray(patch1*255).convert('L').save(fout)
        print('Saved to %s' % fout)

    # coronal
    patch1 = patch[:,patch.shape[1]/2,:]
    plt.imshow(patch1,
               cmap='gray')
    plt.show()
    
    # Save preview
    fout = pth + '_slice1.png'
    if not os.path.isfile(fout):
        Image.fromarray(patch1*255).convert('L').save(fout)
        print('Saved to %s' % fout)
    
    # sagittal
    patch1 = patch[:,:,patch.shape[2]/2]
    plt.imshow(patch1,
               cmap='gray')
    plt.show()
    
    # Save preview
    fout = pth + '_slice2.png'
    if not os.path.isfile(fout):
        Image.fromarray(patch1*255).convert('L').save(fout)
        print('Saved to %s' % fout)
    
    # Save volume
    fout = pth + '.zpkl'
    if not os.path.isfile(fout):
        zipPickle.save({'img':np.uint16(patch*(2**16-1)), 
                        'cand_mm':cand_mm, 
                        'origin_mm':origin_mm,
                        'spacing_mm':spacing_mm,
                        'diameter_mm':diameter_mm},
                       fout)
        print('Saved to %s' % fout)
        
    return 1

#%% Convert - demo
uid2patch(uid0[0], cands_pos)
    
#%% Convert - positive examples
n_to_convert = 3
for row in range(len(cands_pos)):
    cand = cands_pos.loc[row,:]
    if cand2patch(cand):
        n_to_convert -= 1
        if n_to_convert == 0:
            break

#%% Convert positive candidates to patches
for uid1 in uids0:
    uid2patch(uid1, cands_pos)

#%% Convert candidates in given subsets to patches
subset_incl = range(2,10)

for subset1 in subset_incl:
    uids_in_subset1 = uid_subset.seriesuid[ \
            uid_subset.subset.isin([subset1])]
    for uid1 in uids_in_subset1:
        uid2patch(uid1, cands_neg)