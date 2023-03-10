#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:49:37 2021

@author: paco
"""

'''
Debugging 2D_fall_animated
'''
import numpy as np

from matplotlib import pyplot as plt


plt.ion

#%% Lambdas and coversions
to_rad = lambda dg: dg*np.pi/180   #easy formula to express degreen in rads
to_degrees = lambda rad: rad* 180 / np.pi

module = lambda v: (v[0]**2 + v[1]**2)**(1/2)

angle = lambda v: np.arctan(v[1]/v[0]) 

antivector_u = lambda v: -1 * v/module(v)


#%% initial variables

rho = 1.225 #air density 

A = 2.2 #area of the body 1 m2

Cd = 1  #drag coefficient

m = 2300 #100Kg mass
    
v = np.array([1000,0.1]) # vx 2 m/s backwards and 4 m/s upwards

v_mod = module (v)

G = np.array([0,-9.8])*m # G-FORCE 

D_mod = 0.5 * (v_mod**2)*A*Cd*rho

D = antivector_u (v) * D_mod 

R = G + D

a = R/m

s = 0

h = 5000; graph_lim = h #meters

total_time = 0

'''
status = np.array ([v[0],v[1],0,0, h])
status_angles = np.array([0,0]) #to check if D_angle opposite to v_angle
status_R = np.array( [D[0],D[1],R[0],R[1]] )
'''

#%%Preparing plots
#%% Axes definiton

fig = plt.figure('fall analysis', figsize = (6,6))



ax_h = fig.add_subplot(1,1,1)

ax_h.set_title ('Height (m)')#; ax_h.set_xlim([0, graph_lim]); ax_v.set_ylim([0,graph_lim])


#%% Main loop

while h > 10:
    
    ax_h.clear()
    
    t =  0.1 #second
    
    D_mod = 0.5 * (v_mod**2)*A*Cd*rho
    
    angle_v = angle(v)
    
    D = antivector_u(v) * D_mod
    
    angle_D = angle(D)
    #D = np.array([0,0])
    
    R = G+D
    
    a = R / m   #V_MOD NEVER CHANGES, THATŚ WRONG!!!! LOOK FOR THE PROBLEM
    
    v = v  + a * t   #vectorial calculation
    
    v_mod = module (v)
    
    h = h + v[1]*t + 0.5*a[1]*t**2
    
    s = s + v[0]*t + 0.5*a[0]*t**2
    
 '''   
    status = np.vstack ((status, [v[0],v[1],D[0],D[1], h]))
    status_R = np.vstack( (status_R, [D[0],D[1],R[0],R[1]]))
    status_angles = np.vstack((status_angles,[to_degrees(angle_v),to_degrees(angle_D)]))
'''    
    g_s =( module(R)/m ) / 9.8
    
    print ('acceleration in gs: {:.2f}'.format(g_s))
    
    total_time += t
    
    ax_h.set_xlim([-50,graph_lim+50]); ax_h.set_ylim([0,graph_lim+50])
    ax_h.scatter(s,h, color = 'b')
    
    plt.pause(0.1)