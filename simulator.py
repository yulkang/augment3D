#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simulator : simulate CT images and provide segmentation

Created on Fri Oct 14 22:00:58 2016

@author: yulkang
"""

import numpy as np

def main(dens_vessel=3, dens_membrane=3, dens_bronchi=1):
    vol = default_volume()
    
    simulate_vessel(vol, dens_vessel)
    simulate_membrane(vol, dens_membrane)
    simulate_bronchi(vol, dens_bronchi)
    simulate_nodule(vol, dens_nodule)
    
def simulate_vessel(vol, dens, mean_lum, stdev_lum):
    center = rand()
    angle_xy = rand()
    angle_xz = rand()
    dens = np.random.normal(mean_lum, stdev_lum)
    
def simulate_membrane(vol, dens, mean_lum, stdev_lum):
    center = rand()
    angle_xy = rand()
    angle_xz = rand()
    dens = np.random.normal(mean_lum, stdev_lum)

def simulate_membrane(vol, dens, mean_lum, stdev_lum):
    center = rand()
    angle_xy = rand()
    angle_xz = rand()
    dens = np.random.normal(mean_lum, stdev_lum)
    
def simulate_nodule(vol, dens, mean_lum, stdev_lum):
    center = rand()
    angle_xy = rand()
    angle_xz = rand()
    dens = np.random.normal(mean_lum, stdev_lum)
    