#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
MATLAB-style ecdf(sample)

Created on Mon Oct 17 20:38:58 2016

@author: yulkang
"""

def ecdf(sample=None):
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    if sample is None:
        sample = np.random.uniform(0, 1, 50)
        
    f = sm.distributions.ECDF(sample)
    x = np.unique(sample)
    y = f(x)
    plt.step(x, y, where='post')
    plt.ylim((0,1))
    plt.show()