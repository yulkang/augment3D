#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
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

#%%
def load_itk_image(filename):
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
img_path  = 'Data/LUNA/image/1.3.6.1.4.1.14519.5.2.1.6279.6001.231645134739451754302647733304.mhd'
cand_path = 'Data/LUNA/candidates.csv'

        

#%% load image
numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path) 
print numpyImage.shape
print numpyOrigin
print numpySpacing

#%% load candidates
cands = readCSV(cand_path)
print cands

#%% get candidates
for cand in cands[1:]:
    worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
    voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    voxelWidth = 65
    
#%% Convert to patches
for cand in cands[1:]:
    #%% Load corresponding image
    uid = cand[0]
    
    
    worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
    voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    voxelWidth = 65
    
    if (voxelCoord[1] < voxelWidth/2)  \
            or (voxelCoord[1] > numpyImage.shape[2]-voxelWidth/2) \
            or (voxelCoord[2] < voxelWidth/2) \
            or (voxelCoord[2] > numpyImage.shape[2]-voxelWidth/2):
        continue
    
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
