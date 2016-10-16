#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:11:33 2016

@author: yulkang
"""

def dirfiles(pth='.'):
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(pth) if isfile(join(pth, f))]
            

def dirdirs(pth='.'):
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(pth) if not isfile(join(pth, f))]