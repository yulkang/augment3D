#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Loads 3D image files (.mhd) and extracts 3D patches with a margin
so that the patches can be augmented with translation.

Images are compressed in-memory before saving, 
and saved along with .tif previews on XY, YZ, and XZ planes
for an easy sanity check.

Some parts are from SimpleITKTutorial.ipynb, available at:
    https://grand-challenge.org/site/luna16/tutorial/

Created on Sat Oct 15 16:22:52 2016

@author: yulkang
"""

#%% Import
import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import pandas as pd
import warnings

import compare_cand_vs_annot as annot
import paths

#%% Define functions to load images / CSVs
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
    
#%% Normalize the intensity of the image.
maxHU = 400.
minHU = -1000.

def normalizePlanes(npzarray):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray
    
#%% Load candidates.
cands = annot.cands_vs_annot
cands_pos = annot.cands_pos
uid_subset = annot.uid_subset
cands_neg = annot.cands_neg

#%% Get unique uids.
uid0 = cands.seriesuid
uids0 = cands.seriesuid.unique()

#%% Output formats based on the annotated radius of the lesion.
# max nodule radius in the LUNA dataset is 16.14mm.
# So we will divide the range into 0-4, 4-8, and 8-16mm.
radius_min_incl = [0, 4, 8]
radius_max_incl = [4, 8, np.inf]
radius_max = [4, 8, 16]
spacings_output_mm = [0.8, 1.6, 3.2]
radius_out_per_in =  [1.5, 1.5, 1.5] # >1 to have a margin for shifting.
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
    tbl.to_csv(paths.meta_file, sep=',', index=False)
    return tbl
            
def uid2mhd_file(uid):
    return os.path.join(paths.img_dir, uid + '.mhd')

#%% Import uid2meta
if os.path.isfile(paths.meta_file):
    print('meta_file exists. Loading uid2meta from %s' % paths.meta_file)
    tbl = pd.read_csv(paths.meta_file)
else:
    tbl = uid2meta(uids0)
          
#%% Functions to export patches after interpolation
def uid2patch(uid, cands1, 
              n_to_convert=None, 
              output_format=output_formats.iloc[0,:]):
    """
    Export patches given the image ID (uid) and the coordinates (cands1).
    """
    
    # Load the image only if different from previous
    if (uid2patch.uid_prev is None) or \
            (uid != uid2patch.uid_prev):
        
        img_file = os.path.join(paths.img_dir, uid + '.mhd')
        
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
# Initialize a function attribute.
uid2patch.uid_prev = None

def cand2patch(cand, img_np=None, origin_mm=None, spacing_input_mm=None,
               is_annotation=True,
               output_format=output_formats.iloc[0,:]):
    """
    Export a patch given a candidate coordinate.
    """
    
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
    
    if not os.path.isdir(paths.patch_dir):
        os.mkdir(paths.patch_dir)
    
    if os.path.isfile(pth + '.zpkl') and \
            os.path.isfile(pth + '_slice0.png') and \
            os.path.isfile(pth + '_slice1.png') and \
            os.path.isfile(pth + '_slice2.png'):
        print('Exists already. Skipping %s' % pth)
        return 0        
    
    img_file = os.path.join(paths.img_dir, uid + '.mhd')                    
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

        grid0_mm[dim] = temp

    grid_mm = np.transpose(delta_grid_mm + cand_mm)
    
    if 'diamter' in cand.index:
        diameter_mm = float(cand.diamater)
    else:
        diameter_mm = -1
                         
    patch = interpn(grid0_mm, img_np, grid_mm, 
                    bounds_error=False,
                    fill_value=np.nan)
    patch = np.reshape(patch, dia_output_vox)
    patch = normalizePlanes(patch)
    
    print(origin_mm)
    print(patch.shape)
    print(np.min(patch))
    print(np.max(patch))
    
    # Axial
    patch1 = patch[patch.shape[0]/2,:,:]
#    plt.imshow(patch1, 
#               cmap='gray')
#    plt.show()
    
    # Save a preview
    fout = pth + '_slice0.png'
    if not os.path.isfile(fout):
        Image.fromarray(patch1*255).convert('L').save(fout)
        print('Saved to %s' % fout)

    # Coronal
    patch1 = patch[:,patch.shape[1]/2,:]
#    plt.imshow(patch1,
#               cmap='gray')
#    plt.show()
    
    # Save a preview
    fout = pth + '_slice1.png'
    if not os.path.isfile(fout):
        Image.fromarray(patch1*255).convert('L').save(fout)
        print('Saved to %s' % fout)
    
    # Sagittal
    patch1 = patch[:,:,patch.shape[2]/2]
#    plt.imshow(patch1,
#               cmap='gray')
#    plt.show()
    
    # Save a preview
    fout = pth + '_slice2.png'
    if not os.path.isfile(fout):
        Image.fromarray(patch1*255).convert('L').save(fout)
        print('Saved to %s' % fout)
    
    # Save the volume after in-memory compression.
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

def cand_scale2patch_file(cand, 
                          output_format=output_formats.iloc[0,:]):
    """
    Name the patch file given the uid, coordinates, and the scale.
    """
    fmt = output_format
    is_pos = cand['is_pos']

    diameter_mm, _ = output_format2size(fmt, is_pos)
    spacing_mm = fmt['spacing_output_mm']
        
    patch_file = cand2patch_file(cand, 
                                 diameter_mm = diameter_mm,
                                 spacing_mm = spacing_mm)
    return patch_file, diameter_mm, spacing_mm
    
def cand2patch_file(cand, diameter_mm=8*2.5*2, spacing_mm=0.5):
    return os.path.join(
            paths.patch_dir, 
            'patch=' + cand['seriesuid']
            + '+x=' + str(np.round(cand['coordX']*10)/10) 
            + '+y=' + str(np.round(cand['coordY']*10)/10) 
            + '+z=' + str(np.round(cand['coordZ']*10)/10)
            + '+dia=' + str(np.round(diameter_mm))
            + '+spa=' + str(np.round(spacing_mm*10)/10))
        
#%% When run as a script
def main():
    #%% Convert annotated candidates to patches.
    # Give a margin because they will be augmented.
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
    
    #%% Convert positive and Negative candidates in given subsets to patches.
    # Do not give a margin since we won't augment them.
    # (There are enough negative candidates, so augmentation is not necessary.
    #  Positive candidates overlap with annotated candidates,
    #  so they are used not for training but for validation and test.)
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
                
            n_uid_converted += (converted > 0)
            if n_uid_converted >= n_uid_to_convert:
                break
            
        if n_uid_converted >= n_uid_to_convert:
            break                
    
#%%
if __name__ == '__main__':
    main()