#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Mostly follows SimpleITKTutorial.ipynb
but loads corresponding file for each cand,
and fill with 0 if the nodule is too close to the boundary.

Created on Sat Oct 15 16:22:52 2016

@author: yulkang
"""

#%% Tutorial - import
import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tif
from scipy.stats import cumfreq
import pandas as pd

#%%
def load_itk_image(filename):
    if not os.path.isfile(filename):
        return None, None, None
    
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing
    
def readCSV(filename):
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines
    
def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord
    
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
uid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.231645134739451754302647733304'

#%% load candidates
cands = readCSV(cand_file)

#%% Get unique uids
uid0 = list()
for cand in cands[1:]:
    uid0.append(cand[0])
uids = list(set(uid0))

#%% Example
def uid2mhd_file(uid):
    return os.path.join(img_dir, uid + '.mhd')

img_file = uid2mhd_file(uid)
numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_file)
list(numpyImage.shape)
list(numpyOrigin)
list(numpySpacing)

#%% Export metadata (shape, origin, spacing)
def uid2meta(uids=uids):
    with open(meta_file, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['uid', 
                            'shape0', 'shape1', 'shape2',
                            'origin0', 'origin1', 'origin2',
                            'spacing0', 'spacing1', 'spacing2'])
        
        for uid in uids:
            img_file = uid2mhd_file(uid)
            numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_file)
            if numpyImage is None:
                continue
            else:
                row = [uid] + list(numpyImage.shape)
                            + list(numpyOrigin)
                            + list(numpySpacing)
                print(row)
                csvwriter.writerow(row)
            
#%% Export
uid2meta(uids)
             
#%% Examine spacing0

   
#%% Export patches
def uid2patch(uid):
    img_file = os.path.join(img_dir, uid + '.mhd')
    if not os.path.isfile(img_file):
        return 0
    
    numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_file)
    print('Loaded ' + img_file)
    
    for cand in cands[1:]:
        if cand[0] == uid:
            cand2patch(cand, numpyImage, numpyOrigin, numpySpacing)
    
    return 1

def cand2patch(cand, numpyImage, numpyOrigin, numpySpacing):
    worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
    voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    voxelWidth = 65
    voxelDepth = 2
    
    if (voxelCoord[1] - voxelWidth/2 < 0)  \
            or (voxelCoord[1] + voxelWidth/2> numpyImage.shape[2]) \
            or (voxelCoord[2] - voxelWidth/2 < 0) \
            or (voxelCoord[2] + voxelWidth/2 > numpyImage.shape[2]) \
            or (voxelCoord[0] < voxelDepth + 1) \
            or (voxelCoord[0] + voxelDepth > numpyImage.shape[0]):
        return
    
    patch = numpyImage[
        voxelCoord[0],
        voxelCoord[1]-voxelWidth/2:voxelCoord[1]+voxelWidth/2,
        voxelCoord[2]-voxelWidth/2:voxelCoord[2]+voxelWidth/2]
    patch = normalizePlanes(patch)
    print 'data'
    print worldCoord
    print voxelCoord
    print patch
    outputDir = 'Data/patches/'
    plt.imshow(patch, cmap='gray')
    plt.show()
    
    pth = os.path.join(
            outputDir, 
            'patch_' + str(worldCoord[0]) 
            + '_' + str(worldCoord[1]) 
            + '_' + str(worldCoord[2]) + 
            '.tiff')
    Image.fromarray(patch*255).convert('L').save(pth)

#%% Convert candidates to patches
for uid in uids:
    uid2patch(uid)
