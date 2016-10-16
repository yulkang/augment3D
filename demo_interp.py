#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
demo_interp

Created on Sun Oct 16 10:45:03 2016

@author: yulkang
"""

#%% Demo - interp1d
import scipy.interpolate as interp
x0 = np.arange(5)
v0 = np.linspace(6,10,5)
x1 = np.linspace(3,6,5)
f = interp.interp1d(x0, v0, bounds_error=False, fill_value=0)
print(f(x1))
   
#%% Demo - interpn
import scipy.interpolate as interp
x0 = [np.arange(2), np.arange(3), np.arange(4)]
v0 = np.random.rand(2,3,4)
x1 = np.transpose(np.array([[1,2,3],[2,3,4]]))
v1 = interp.interpn(tuple(x0), v0, tuple(x1), bounds_error=False, fill_value=0)
print(v1)
   
