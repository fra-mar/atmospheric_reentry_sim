#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:39:21 2021

@author: paco
"""

'''plotting surfaces in pyplot'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-1,1,4)
y = np.linspace(-1,1,4)
z = y.copy()

X,Y = np.meshgrid(x,y)
Z=0*X + 0*Y #+ 1.0964608113924048

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('lat')

surf1 = ax.plot_surface(X, Y, Z, alpha = 0.3)

YY,ZZ = np.meshgrid ( y,z)
XX = 0*Y + 0*Z
surf2 = ax.plot_surface (XX,YY,ZZ, alpha = 0.3)