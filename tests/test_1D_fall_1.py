#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:59:24 2021

@author: paco
"""

'''falling ball, assuming gravity and drag.
Drag assumes constant air density and drag coefficient'''

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import sleep

#%% Defining start tvariables

m, area = 2.3e3, 2.2  #mass and front area 

rho = 1.225 #Kg/m3 air density of air 15C and 100KPa

drag_coef = 1.0  #dimensionsless

a_g = 9.8  #g acceleration

v0, v = 0, 0

t0 = 0

h0 = 1e4
h = h0

interval = 0.01

status = np.array ( [0,0,v0,h0] ) #array that stores time,, drag, speed, height


#%% main loop

count = 0

while h >0 :
    
    t0 = dt.datetime.now()
    
    sleep(0.2)
    
    t1 = dt.datetime.now()
    
    dif_t = t1-t0
    
    t = dif_t.seconds + dif_t.microseconds / 1000000
    
    G = -m * a_g
    
    D= 0.5 * area * v**2 * drag_coef*rho
    
    R= D + G   #resultant force R
    
    a = R/m #resultant acceleration
    
    v = v0 + a * t
    
    h = h0 + v0*t + 0.5 *a * t**2
    
    h0, v0 = h, v
    
    print ('t: {:.2f}s\t Drag: {:.2f}N\t Vel:{:.2f} m/s\t h: {:.2f} m'.format(t*count,D,v,h) )
    
    status = np.vstack ((status, [t*count, D, v, h] ))
    
    count += 1
    
#%% Plotting 
    
fig = plt.figure('fall analysis', figsize = (10,7))

fig.suptitle ('Falling body mass = {}Kg, A = {}m2, Cd = {}'.format(m,area,drag_coef))

ax_v = fig.add_subplot(2,2,1)

ax_v.plot (status[:,2], label = 'speed, m/s')

ax_D = fig.add_subplot(2,2,2)

ax_D.plot (status[:,1], label = 'drag, N')

ax_h = fig.add_subplot(2,1,2)

ax_h.plot (status[:,3], label = 'height, m')

ax_v.set_title ('Vertical speed m/s')
ax_D.set_title ('Drag (N)')
ax_h.set_title ('Height (m)')

plt.show()


