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

from pynput import keyboard


plt.ion

#%% Lambdas and coversions

to_rad = lambda dg: dg*np.pi/180   #easy formula to express degreen in rads

to_degrees = lambda rad: rad* 180 / np.pi

module = lambda v: (v[0]**2 + v[1]**2)**(1/2)

angle = lambda v: np.arctan(v[1]/v[0]) 

antivector_u = lambda v: -1 * v/module(v)

ortovector_u = lambda v: np.array([ -v[1],v[0]]) /module (v)

#%% Controlls
def on_press(key):
    
    global rot
    if key.char == 'a':
        rot = 1
    elif key.char == 'z':
        rot = -1
    else:
        pass
    print (str(rot),'\n')

listener=keyboard.Listener(on_press=on_press)
listener.start()  

#%% initial variables

rho = 1.225 #air density 

A = 2.2 ; m = 2300; rot = 1 #100Kg mass #area of the body 1 m2 rot = flying upward (0 upside-down)

Cd = 1  #drag coefficient

A_lift = 0.5  #area of the surface intended as wing

Cl = 1.2   #lift coefficient
    
v = np.array([1000,0.1]) # vx 2 m/s backwards and 4 m/s upwards

v_mod = module (v)

G = np.array([0,-9.8])*m # G-FORCE 

D_mod = 0.5 * (v_mod**2)*A*Cd*rho

D = antivector_u (v) * D_mod 

L_mod = 0.5 * (v_mod**2)*A_lift*Cl*rho

L = ortovector_u (v) * L_mod * rot

R = G + D + L

a = R/m

s = 0

h = 10000; graph_lim = h #meters

total_time = 0


status = np.array ([v[0],v[1],0,0, h])
status_angles = np.array([0,0]) #to check if D_angle opposite to v_angle
status_DL = np.array( [D[0],D[1],L[0],L[1]] )


def gravity_calc (h):
    global m
    grav_constant = 6.67408*1e-11 #units m3 kg-1 s-2
    radius_earth = 6.371*1e6   #units in m
    earth_mass = 5.9722*1e24 #units in kg
    Gy = grav_constant * earth_mass * m / ( (radius_earth + h)**2 )
    G = np.array( [0,-Gy])
    return G
    
#%% Axes definiton

fig = plt.figure('fall analysis', figsize = (6,6))



ax_h = fig.add_subplot(1,1,1)

ax_h.set_title ('Height (m)')#; ax_h.set_xlim([0, graph_lim]); ax_v.set_ylim([0,graph_lim])


#%% Main loop

while h > 10:
    
    ax_h.clear()
    
    t =  0.1 #second
    
    D_mod = 0.5 * (v_mod**2)*A*Cd*rho   #calculates drag
    
    D = antivector_u(v) * D_mod         
    
    L_mod = 0.5 * (v_mod**2)*A_lift*Cl*rho    #calculates lift

    L = ortovector_u (v) * L_mod * rot
    
    G = gravity_calc(h)                  #calculates gravity as function of h
    
    R = G + D + L                        #resultant force
    
    a = R / m   
        
    v = v  + a * t                      #vectorial calculation
    
    v_mod = module (v)
    
    h = h + v[1]*t + 0.5*a[1]*t**2
    
    s = s + v[0]*t + 0.5*a[0]*t**2
    
    
    status = np.vstack ((status, [v[0],v[1],L[0],L[1], h]))
    status_DL = np.vstack( (status_DL, [D[0],D[1],L[0],L[1]]))
    
        
    g_s =( module(R)/m ) / 9.8
    
    #print ('acceleration in gs: {:.2f}  Gy = {:.1f}'.format(g_s, G[1]))
    
    print ('Vx {:.1f} Vy {:.1f}    Lx {:.1f} Ly {:.1f}'.format(v[0],v[1],L[0],L[1]))
    
    total_time += t
    
    ax_h.set_xlim([-50,graph_lim+1000]); ax_h.set_ylim([0,graph_lim+2000])
    ax_h.scatter(s,h, color = 'b')
    
    plt.pause(0.1)