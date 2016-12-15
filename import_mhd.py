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
from pysy.stat import ecdf

#%% Define functions
def load_itk_image(filename):
    if not os.path.isfile(filename):
        return None, None, None
    
    try:
        itkimage = sitk.ReadImage(filename)
        img_np = sitk.GetArrayFromImage(itkimage)
        origin_mm = np.array(list(reversed(itkimage.GetOrigin())))
        spacing_input_mm = np.array(list(reversed(itkimage.GetSpacing())))
        return img_np, origin_mm, spacing_input_mm
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
img_dir = '/Volumes/YK_SSD_1TB/LUNA/all'
#patch_dir = 'Data/patches/'
patch_dir = '/Volumes/YK_SSD_1TB/LUNA/patches'

subset_file = 'Data/LUNA/uid_subset.csv'
meta_file = 'Data/LUNA/img_meta.csv'

## an example uid that is in subset0
#uid0 = '1.3.6.1.4.1.14519.5.2.1.6279.6001.213140617640021803112060161074'

#%% load candidates
cands = annot.cands_vs_annot
cands_pos = annot.cands_pos
uid_subset = annot.uid_subset
cands_neg = annot.cands_neg

#%% Get unique uids
uid0 = cands.seriesuid
uids0 = cands.seriesuid.unique()

#%% Output format based on radius
# max radius is 16.14mm
radius_min_incl = [0, 4, 8]
radius_max_incl = [4, 8, np.inf]
radius_max = [4, 8, 16]
spacings_output_mm = [0.8, 1.6, 3.2] # [0.4, 0.8, 1.6]
radius_out_per_in =  [1.5, 1.5, 1.5] # [  2,   2,   2]
output_formats = pd.DataFrame({'radius_min_incl': radius_min_incl, 
                               'radius_max_incl': radius_max_incl,
                               'radius_max': radius_max,
                               'spacing_output_mm': spacings_output_mm,
                               'radius_out_per_in': radius_out_per_in})

def output_format2size(output_format, is_pos, 
                       margin=1):
    if is_pos:
        radius_mm = output_format['radius_max'] \
                           * (output_format['radius_out_per_in'] \
                              + margin)
    else:
        radius_mm = output_format['radius_max'] \
                                 * output_format['radius_out_per_in']
            
    diameter_mm = radius_mm * 2
    spacing_mm = output_format['spacing_output_mm']
    diameter_vox = np.int32(diameter_mm / spacing_mm)
    
    return diameter_mm, diameter_vox

#%% uid2meta - example
def uid2mhd_file(uid):
    return os.path.join(img_dir, uid + '.mhd')

#%% Export metadata (shape, origin, spacing)
def uid2meta(uids=uids0):
    tbl = pd.DataFrame(columns=['uid',
                                'shape0', 'shape1', 'shape2',
                                'origin0', 'origin1', 'origin2',
                                'spacing0', 'spacing1', 'spacing2'])
    tbl.uid = uids
    print(tbl.head())
    
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
          
#%% Functions to export patches after interpolation
def uid2patch(uid, cands1, 
              n_to_convert=None, 
              output_format=output_formats.iloc[0,:]):
    if (uid2patch.uid_prev is None) or \
            (uid != uid2patch.uid_prev):
        
        img_file = os.path.join(img_dir, uid + '.mhd')
        
        if not os.path.isfile(img_file):
            return 0
        
        img_np, origin_mm, spacing_mm = load_itk_image(img_file)
        
        if img_np is not None:
            print('Loaded ' + img_file)
            uid2patch.uid_prev = uid
            uid2patch.img_np_prev = img_np
            uid2patch.origin_mm_prev = origin_mm
            uid2patch.spacing_mm_prev = spacing_mm
    else:
        img_np = uid2patch.img_np_prev
        origin_mm = uid2patch.origin_mm_prev
        spacing_mm = uid2patch.spacing_mm_prev
    
    row_incl = np.nonzero(cands1.seriesuid == uid)[0]
    
    if n_to_convert is None:
        n_to_convert = np.inf
    n_converted = 0

    for i_row in row_incl:
        n_converted += cand2patch(cands1.iloc[i_row], 
                                  img_np, origin_mm, spacing_mm,
                                  output_format=output_format)
        if n_converted >= n_to_convert:
            break # DEBUG
    
    return n_converted
uid2patch.uid_prev = None

def cand_scale2patch_file(cand, 
                          output_format=output_formats.iloc[0,:]):
    fmt = output_format
#    print(cand) # DEBUG
    is_pos = cand['is_pos']

    diameter_mm, _ = output_format2size(fmt, is_pos)
    spacing_mm = fmt['spacing_output_mm']
        
    patch_file = cand2patch_file(cand, 
                                 diameter_mm = diameter_mm,
                                 spacing_mm = spacing_mm)
    return patch_file, diameter_mm, spacing_mm
    
def cand2patch_file(cand, diameter_mm=8*2.5*2, spacing_mm=0.5):
    return os.path.join(
            patch_dir, 
            'patch=' + cand['seriesuid']
            + '+x=' + str(np.round(cand['coordX']*10)/10) 
            + '+y=' + str(np.round(cand['coordY']*10)/10) 
            + '+z=' + str(np.round(cand['coordZ']*10)/10)
            + '+dia=' + str(np.round(diameter_mm))
            + '+spa=' + str(np.round(spacing_mm*10)/10))
    
def cand2patch(cand, img_np=None, origin_mm=None, spacing_input_mm=None,
               is_annotation=True,
               output_format=output_formats.iloc[0,:]):
#               max_radius=8, 
#               spacing_output_mm=0.5,
#               radius_out_per_in=2,
#               radius_margin=8
#               ):
    from scipy.interpolate import interpn
    from pysy import zipPickle
    
    uid = cand['seriesuid']
    cand_mm = np.reshape(
        np.asarray([cand['coordZ'],
                    cand['coordY'],
                    cand['coordX']]),
        (3,1)) # z, y, x; bottom->up, ant->post, right->left
                    
    uid = cand['seriesuid']
    pth, dia_output_mm, spacing_output_mm = cand_scale2patch_file(
            cand, output_format)
    
    spacing_output_mm = np.zeros(3) + spacing_output_mm
    dia_output_mm = np.zeros(3) + dia_output_mm
    dia_output_vox = dia_output_mm / spacing_output_mm
    
    if not os.path.isdir(patch_dir):
        os.mkdir(patch_dir)
    
    if os.path.isfile(pth + '.zpkl') and \
            os.path.isfile(pth + '_slice0.png') and \
            os.path.isfile(pth + '_slice1.png') and \
            os.path.isfile(pth + '_slice2.png'):
        print('Exists already. Skipping %s' % pth)
        return 0        
    
    img_file = os.path.join(img_dir, uid + '.mhd')                    
    if img_np is None:
        if not os.path.isfile(img_file):
            return 0
        
        img_np, origin_mm, spacing_input_mm = load_itk_image(img_file)
        print('Loaded ' + img_file)
    
    delta_grid_mm = np.mgrid[
        -dia_output_mm[0]/2:dia_output_mm[0]/2:spacing_output_mm[0],
        -dia_output_mm[1]/2:dia_output_mm[1]/2:spacing_output_mm[1],
        -dia_output_mm[2]/2:dia_output_mm[2]/2:spacing_output_mm[2]]
    delta_grid_mm = np.reshape(delta_grid_mm, (3,-1))
        
    grid0_mm = [None, None, None]
    for dim in range(3):
        temp = np.arange(img_np.shape[dim]) \
                * spacing_input_mm[dim] \
                + origin_mm[dim]

#        print(grid0_mm[dim]) # DEBUG
#        print(temp)

        grid0_mm[dim] = temp

    grid_mm = np.transpose(delta_grid_mm + cand_mm)
    
    if 'diamter' in cand.index:
        diameter_mm = float(cand.diamater)
    else:
        diameter_mm = -1
                         
#    print(cand)
#    print(grid0_mm)
#    print(grid_mm.shape)
#    print(grid_mm)
    
    patch = interpn(grid0_mm, img_np, grid_mm, 
                    bounds_error=False,
                    fill_value=np.nan)
    patch = np.reshape(patch, dia_output_vox)
    patch = normalizePlanes(patch)
    
    print(origin_mm)
    print(patch.shape)
    print(np.min(patch))
    print(np.max(patch))
    
    # axial
    patch1 = patch[patch.shape[0]/2,:,:]
#    plt.imshow(patch1, 
#               cmap='gray')
#    plt.show()
    
    # Save preview
    fout = pth + '_slice0.png'
    if not os.path.isfile(fout):
        Image.fromarray(patch1*255).convert('L').save(fout)
        print('Saved to %s' % fout)

    # coronal
    patch1 = patch[:,patch.shape[1]/2,:]
#    plt.imshow(patch1,
#               cmap='gray')
#    plt.show()
    
    # Save preview
    fout = pth + '_slice1.png'
    if not os.path.isfile(fout):
        Image.fromarray(patch1*255).convert('L').save(fout)
        print('Saved to %s' % fout)
    
    # sagittal
    patch1 = patch[:,:,patch.shape[2]/2]
#    plt.imshow(patch1,
#               cmap='gray')
#    plt.show()
    
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
                        'spacing_input_mm':spacing_input_mm,
                        'diameter_mm':diameter_mm},
                       fout)
        print('Saved to %s' % fout)
        
    return 1

#%% When run as a script
def main():
    #    #%% Test load_itk_image
    #    img_file = uid2mhd_file(uid0[0])
    #    img_np, origin_mm, spacing_input_mm = load_itk_image(img_file)
    #    print(img_np.shape)
    #    print(origin_mm)
    #    print(spacing_input_mm)
    #    
    #    #%% uid2patch demo
    #    uid2patch(uid0[0], cands_pos)
    #        
    #    #%% cand2patch demo
    #    n_to_convert = 3
    #    n_converted = 0
    #    for row in range(len(cands_pos)):
    #        cand = cands_pos.loc[row,:]
    #        n_converted += cand2patch(cand)
    #        if n_converted >= n_to_convert:
    #            break
    
    #%% Convert
    # Annotated (positive) candidates to patches
    n_uid_to_convert = np.inf
    n_uid_converted = 0
    n_uid = len(uids0)
    for i_uid in range(n_uid):
        uid1 = uids0[i_uid]
        print('uid %d/%d' % (i_uid, n_uid))

        for ii in range(len(output_formats)):
            fmt = output_formats.iloc[ii,:]    
    
            converted = uid2patch(uid1, cands_pos, 
                                  output_format=fmt)
#                      max_radius = fmt.radius_max,
#                      spacing_output_mm = fmt.spacing_output_mm,
#                      radius_out_per_in = fmt.radius_out_per_in,
#                      radius_margin = fmt.radius_max)
        
        n_uid_converted += (converted > 0)
        if n_uid_converted >= n_uid_to_convert:
            break
        
    #%%
    # Positive and Negative candidates in given subsets to patches
    # Do not give margin since we won't augment the data
    subset_incl = [0] # range(10)
    n_subset = len(subset_incl)
    n_uid_to_convert = np.inf
    n_uid_converted = 0
    
    for i_subset in range(n_subset):
        subset1 = subset_incl[i_subset]
        uids_in_subset1 = uid_subset.seriesuid[ \
                uid_subset.subset.isin([subset1])]

        n_uid = len(uids_in_subset1)            
        for i_uid in range(n_uid):
            uid1 = uids_in_subset1[i_uid]
            print('uid %d/%d, subset %d' % (i_uid, n_uid, subset1))
            
            for ii in range(len(output_formats)):
                fmt = output_formats.iloc[ii,:]
    
                converted = uid2patch(
                         uid1, cands, 
                         output_format=fmt)
#                         max_radius = fmt.radius_max,
#                         spacing_output_mm = fmt.spacing_output_mm,
#                         radius_out_per_in = fmt.radius_out_per_in,
#                         radius_margin = 0,
#                         n_to_convert = np.inf)
                
            n_uid_converted += (converted > 0)
            if n_uid_converted >= n_uid_to_convert:
                break
            
        if n_uid_converted >= n_uid_to_convert:
            break                
    
#%%
if __name__ == '__main__':
    main()