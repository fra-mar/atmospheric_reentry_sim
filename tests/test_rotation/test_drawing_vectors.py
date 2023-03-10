#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:12:59 2021

@author: paco
"""

'''How to draw vectors
from https://www.geeksforgeeks.org/quiver-plot-in-matplotlib/
'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# Creating arrow
x_pos = [0, 0]   #first number corresponds 1 vector. 2 numbers? will draw 2 vectors
y_pos = [0, 0]
x_direct = [1, 0]
y_direct = [1, -1]

# Creating plot
fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver(x_pos, y_pos, x_direct, y_direct,
		scale = 5)   #quiver gets 4 arguments: an origin, a direction. In this case a scale to match the axis scale

ax.axis([-1.5, 1.5, -1.5, 1.5])

# show plot
plt.show()
