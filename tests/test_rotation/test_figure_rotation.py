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

origin = np.array([0,0])

figure = np.array([[0.0,3],[-1,2],[-1,-1],[-2,-2],[2,-2],[1,-1],[1.0,2],[0,3]])

#figure = np.array( [ [0.0,3.0] , [-3.0,0.0], [3.0,0.0] , [0.0,3.0]  ])
figure_rotated = np.zeros(figure.shape)


def find_coords(fig, orig):
    
    coord_x = [x + orig[0] for x in fig[:,0] ]
    coord_y = [y + orig[1] for y in fig[:,1] ]

    return coord_x,coord_y
#%%Plotting



#%% rotating

theta = 300 * np.pi/180 #degrees converted to radians


for i in range (0, len(figure)):

    x , y = figure[i][0], figure[i][1]
    figure_rotated[i][0] = x*np.cos(theta) - y*np.sin(theta)
    figure_rotated[i][1] = x*np.sin(theta) + y*np.cos(theta)
    
#%% more plotting    
plt.style.use('seaborn')

fig = plt.figure('testing figure rotation', figsize = (6,6))
ax = fig.add_subplot(1,1,1)

figures = [figure, figure_rotated]

for f in figures:

    coord_x, coord_y = find_coords (f, origin)
    ax.plot(coord_x, coord_y)
    


ax.scatter(origin[0],origin[1], marker = 'o')
ax.axis([-6,6,-6,6]) #limits for both axis
ax.set_aspect('equal')

plt.show()


