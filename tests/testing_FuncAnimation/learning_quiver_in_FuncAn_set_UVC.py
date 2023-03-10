#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:33:12 2021

@author: paco
Copied from 
https://www.tutorialspoint.com/plotting-animated-quivers-in-python-using-matplotlib
to learn how to use .set(u,v)
"""
import numpy as np
#import random as rd
from matplotlib import pyplot as plt, animation

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

#x, y = np.mgrid[:2 * np.pi:10j, :2 * np.pi:5j] Original code
x, y = 3., 3. #here I simplify to get a single arrow
u = np.cos(x)
v = np.sin(y)

fig, ax = plt.subplots(1, 1)
#qr = ax.quiver(x, y, u, v, color='red')# original code
qr = ax.quiver(x, y, u, v, label = 'arrow', scale = 5, color='red')
ax.legend ( loc=1)

def animate(num):
   global qr, x, y #...so interesting...doesn't need to get these into de function..
   u = np.cos(x + num * 0.1)
   v = np.sin(y + num * 0.1)

   qr.set_UVC(u, v) #must be an np.array, seems to happen
   #qr.set_color((rd.random(), rd.random(), rd.random(), rd.random()))
   return qr,

anim = animation.FuncAnimation(fig, animate,
                              interval=20, blit=False)
'''
anim = animation.FuncAnimation(fig, animate, fargs=(qr, x, y),
                              interval=50, blit=False)
'''
plt.show()