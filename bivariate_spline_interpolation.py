#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:34:19 2025

@author: josh
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt



n=0.005

x_values = np.array([0,1,2,3,4])
y_values = np.array([0,1,2])
z_values = np.array([
    [1-2*n, 1-4*n, 1-5*n, 1-3*n, 1-2*n],
    [1-n, 1-2*n, 1-3*n, 1-2*n, 1-n],
    [1-0.5*n, 1-0.5*n, 1-0.5*n, 1-0.5*n, 1-0.5*n] ])



interpolation_function = scipy.interpolate.RectBivariateSpline(x_values, y_values, z_values.T, ky = 2)

print(interpolation_function)

x_temp = np.linspace(0, 4, 200)
y_temp = np.linspace(0, 2, 100)
z_meshed = interpolation_function(x_temp, y_temp).T

z_meshed = np.clip(z_meshed, 0, 1)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x_temp, y_temp)
ax.plot_surface(X, Y, z_meshed, cmap='viridis', edgecolor='none', alpha=0.8)

for j, x in enumerate(x_values):
    for k, y in enumerate(y_values):
        ax.scatter(x, y, z_values[k, j], color='red', s=50)

ax.set_title("Interpolated Surface")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_zlim(0, 1.1)
ax.set_box_aspect([4,2,1]) 

plt.plot()
