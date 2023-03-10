#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 10:49:39 2021

@author: paco
"""

'''
testing numpy.meshgrid
from https://www.geeksforgeeks.org/numpy-meshgrid-function/
'''

from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0,5,6)    #shape (6,)

y = np.linspace(0,3,4)    #shape (4,)

x1, y1 = np.meshgrid(x,y)  #now x1.shape (4,6), 6 columns but 4 rows, like y. way around with y1

print (x1); print (y1)

fig = plt.figure(figsize = (12,5))
ax1 = fig.add_subplot (121)
ax1.scatter (x1,y1)

#the output of meshgrid can be used to plot functions

ellipse =  x1**2 + 4*y1**2 
my_function = x1+ (1/(y1+0.1))

ax2 = fig.add_subplot (122)
ax2.contourf(x1, y1, my_function, cmap = 'jet')



plt.show()