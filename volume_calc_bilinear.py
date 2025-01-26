#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:52:49 2025

@author: josh
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

R = np.array([[ 1, 0, 0, 0],
              [-1, 1, 0, 0],
              [-1, 0 ,1, 0],
              [1, -1, -1, 1]])

x_values = np.array([0, 0.25, 0.5, 0.75, 1.0])
y_values = np.array([0, 0.25, 0.5, 0.75, 1.0])
z_values = np.array([
    [0.7, 0.7, 0.7, 0.7],
    [0.8, 0.8, 0.8, 0.8],
    [0.9, 0.9, 0.9, 0.9],
    [1.0, 1.0, 1.0, 1.0]])

def bilinear_interpolation(x00, x10, x01, x11):
    return R.dot(np.array([x00, x10, x01, x11]).T) #a=R*f

def volume_calc(H):
    m, n = H.shape
    # print(H)
    # print('')
    # print(H[:, 1:n-1])
    # print('')
    # print(H[1:m-1, :])
    # print('')
    # print(H[1:m-1, 1:n-1])
    
    return (m-1)*(n-1) - 0.25*(H.sum() + H[:, 1:n-1].sum() + H[1:m-1, :].sum() + H[1:m-1, 1:n-1].sum())
    
   # return( m*n - 0.25(H.sum() + H[]))
    
    
    
print(volume_calc(0.5*np.ones((2,4))))






    
    
    