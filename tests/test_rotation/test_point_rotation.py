#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 44:37 2021

@author: paco
"""

'''
From https://www.khanacademy.org/computing/pixar/sets/rotation/v/sets-8
Test equation for rotational-translation of a single point based on equation:
x' = x cos(theta) - y sin(theta)
y' = x sin(theta) + y cos(theta)

'''
from matplotlib import pyplot as plt

import numpy as np
#%%

x, y = 1, 0    #coordinates for my point

theta = 45 * np.pi/180 #degrees converted to radians

x_p = x*np.cos(theta) - y*np.sin(theta)

y_p = x*np.sin(theta) + y*np.cos(theta)

#%% plotting
plt.style.use('seaborn')

fig = plt.figure('testing point rotation-translation', figsize = (6,6))
ax = fig.add_subplot(1,1,1)

ax.scatter([x, x_p], [y, y_p])

ax.quiver (x,y,x_p-x, y_p-y) #show a vector (originx,or_y,direction_x,dir_y)


ax.axis([-3,5,-3,5]) #limits for both axis
ax.set_aspect('equal')

plt.show()


