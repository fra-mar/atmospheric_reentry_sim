#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:59:06 2021

@author: paco
"""

'''
Creating an visualizing planes with matplotlib and pyrr
'''

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import pyrr

#%%
# create the figure
fig = plt.figure()

# add axes
ax = fig.add_subplot(111,projection='3d')

xx, yy = np.meshgrid(range(10), range(10))
z = (5 - xx - yy) / 2 

# plot the plane
ax.plot_surface(xx, yy, z, alpha=0.5)

plt.show()

normal = pyrr.plane.normal([xx,yy,9])




