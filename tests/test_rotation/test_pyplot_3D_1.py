#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 06:55:18 2021

@author: paco
"""

'''
tesing matplotlib 3D
https://www.geeksforgeeks.org/three-dimensional-plotting-in-python-using-matplotlib/


from matplotlib import pyplot as plt

fig = plt.figure()
ax = plt.axes(projection = '3d') #THIS IS ALL YOU NEED TO START WITH!!

'''

#%% More properly done..you need 3D toolkit

# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')
'''
# defining all 3 axes
z = np.linspace(0, 1, 100)
x = z * np.sin(25 * z)
y = z * np.cos(25 * z)


# plotting
ax.plot3D(x, y, z, 'green')

#scatter

c = x + y
ax.scatter(x+0.1, y+0.1, z, c = c)
'''
v_vel = np.array([[0.5547002 , 0.83205029, 0.        ]])
L_u = np.array([[0.0],[0.09632268],[0.54627305]]).reshape((1,3))  
ax.quiver (0,0,0,  v_vel[0][0], v_vel[0][1], v_vel[0][2], color = 'black', length = 0.05)
ax.quiver (0,0,0,  2,3,0, color = 'blue', length = 0.05)

ax.set_title('3D line plot geeks for geeks')
plt.show()
'''
#%% surfaces
# see more surface alternatives in https://www.geeksforgeeks.org/three-dimensional-plotting-in-python-using-matplotlib/



# function for z axis
def f(x, y):
	return np.sin(np.sqrt(x ** 2 + y ** 3))

# x and y axis
x = np.linspace(-1, 5, 10)
y = np.linspace(-1, 5, 10)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)


# a wireframe
fig3 = plt.figure()
ax3 = plt.axes(projection ='3d')
ax3.plot_wireframe(X, Y, Z, color ='green')
ax3.set_title('wireframe geeks for geeks');

# same image but with contours
fig2 = plt.figure()
ax2 = plt.axes(projection ='3d')

# ax.contour3D is used plot a contour graph
ax2.contour3D(X, Y, Z)
'''

