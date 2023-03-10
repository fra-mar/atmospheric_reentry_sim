#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:49:37 2021

@author: paco
"""

'''
In this version....
linear transformation has been a bit painfull to get a correct lift vector
Now I try another idea. The lift vector is perpendicular to vel_after but
between vel_after and z_u. I try that propierty to calculate it
Based partially on https://math.stackexchange.com/questions/496662/how-to-find-a-perpendicular-vector
'''
import numpy as np

from matplotlib import pyplot as plt

from mpl_toolkits import mplot3d

from pynput import keyboard

from lift import  lift_vector

from lin_transf_matrix_1 import lift_function_beta, find_angle


#%% Lambdas and coversions

to_rad = lambda dg: dg*np.pi/180   #easy formula to express degreen in rads

to_degrees = lambda rad: rad* 180 / np.pi

module = lambda v: np.sqrt( np.sum(v)**2) 

antivector_u = lambda v: -1 * v/module(v)

v_unitary = lambda v: v/module(v)




#%% initial variables

rho = 1.225 #air density 

A = 2.2 ; m = 2300.0; rot = 1 #100Kg mass #area of the body 1 m2 rot = flying upward (0 upside-down)

Cd = 1  #drag coefficient

A_lift = 0.5  #area of the surface intended as wing

Cl = 1.2   #lift coefficient
    
vel_init = np.array([[1.01,100.01,1.01]]) # vx 2 m/s backwards and 4 m/s upwards

vel_after = np.array([[1.01,100.01,-10.01]])

vel_aux = vel_init #this will help to prevent runtime error when calculating lift vector

vel_original = vel_init #reference vector to calculate linear transformation matrix

v_lift_original = np.cross(vel_after,   np.cross(vel_original,vel_after)    )

theta = 0 #angle in degrees. angle intended to turn the lift vector about velocity axis

x,y = 0,0 # x NS axis, y WE axis...in a 3D aixs

h = 10000.01; graph_lim = h #meters #h similar to z ina 3D axis

total_time = 0

status = np.array ([y,x, h])
status_forces = np.array ( [0,0,0] )
status_angle = np.array ( [0])

rot, act = 0, 0

def gravity_calc (h):
    global m
    grav_constant = 6.67408*1e-11 #units m3 kg-1 s-2
    radius_earth = 6.371*1e6   #units in m
    earth_mass = 5.9722*1e24 #units in kg
    Gy = grav_constant * earth_mass * m / ( (radius_earth + h)**2 )
    G = np.array( [[0,0,-Gy]])
    return G
    
#%% start vectors and angles
x_u= np.array( [[1,0.0,0.0]])

y_u = np.array( [[0.,1.,0.]])

z_u = np.array( [[0.,0.,1]])

alpha, beta, gamma = 0,0,0

#%% Axes definiton

plt.ion()

fig = plt.figure('fall analysis', figsize = (12,6))

ax_xy = fig.add_subplot(1,2,2) #ax for surface, bird view

#ax_h = fig.add_subplot(1,2,1)

ax_vectors =  fig.add_subplot(121,projection = '3d')

ax_vectors.view_init(20,30)

#%% Controlls

def on_press(key):
    
    global rot
    if key.char == 'a':
        rot = 1 
    if key.char == 'z':
        rot = -1
    else:
        pass
    print (str(rot),'\n')

listener=keyboard.Listener(on_press=on_press)
listener.start()  

#%% Main loop

while h > 10:
    
    #ax_h.clear()
    ax_xy.clear()
    ax_vectors.clear()
    
    if rot == 1 or rot == -1:
        print ('yep')
        if rot == 1:
            theta += np.deg2rad(1.5)
        elif rot == -1:
            theta -=np.deg2rad(1.5)
        
        print (rot,np.rad2deg(theta) )
    
    t =  0.1 #second
    
    v_mod = module (vel_init)
    
    D_mod = 0.5 * (v_mod**2)*A*Cd*rho   #calculates drag
    
    D = antivector_u(vel_init) * D_mod         
    
    L_mod = 0.5 * (v_mod**2)*A_lift*Cl*rho    #calculates lift
    
    v_lift_u, v_lift_r_u = lift_vector (vel_after, theta)
    
    L = v_lift_r_u * L_mod 
    
    G = gravity_calc(h)                  #calculates gravity as function of h
    
    #L = np.array( [[0.,0.,0.]])
    
    R = G + D + L                        #resultant force
    
    a = R / m   
        
    vel_after = vel_init  + a * t                      #vectorial calculation
    
    h = h + vel_after[0][2]*t + 0.5*a[0][2]*t**2  
    
    x = x + vel_after[0][0]*t + 0.5*a[0][0]*t**2
    
    y = y + vel_after[0][1]*t + 0.5*a[0][1]*t**2
    
    g_s =( module(R)/m ) / 9.8
    
    status = np.vstack ((status, [y,x, h]))
    
    status_forces = np.vstack( (status_forces, [D_mod, L_mod, g_s]) )
    
    status_angle = np.vstack ( (status_angle, [np.rad2deg( find_angle ( L, vel_after))]) )
    
    total_time += t
    
    ax_xy.set_title ('West to East, North/South')
    
    ax_xy.set_xlim([-100,graph_lim/3]); ax_xy.set_ylim([-graph_lim/4,graph_lim/4])
    
    ax_xy.scatter(y, x, color = 'b')
        
    
    altitude_label = 'Altitude {:.2f} m'.format(h)
    ground_speed = 'GS {:.2f} m/s'.format ( module ( np.array([vel_after[0][0],vel_after[0][1]])))
    vertical_speed = 'VS {:.2f} m/s'.format (vel_after[0][2])
    ax_xy.text(1,2000, altitude_label)
    ax_xy.text(1,1800, vertical_speed)
    ax_xy.text(1,1600, ground_speed)
    ax_xy.grid()
    ax_vectors.scatter([-3,3,0,0,0,0],[0,0,-3,3,0,0],[0,0,0,0,3,-3])
    ax_vectors.set_xlabel('x')
    ax_vectors.set_ylabel('y')
    ax_vectors.set_zlabel('z')
    
    R_u, G_u, D_u = v_unitary(R), v_unitary(G),  v_unitary(D)
    vectors = [ (R_u,'red') , (D_u,'green') ,
               (G_u, 'black'), (v_lift_r_u, 'blue')]
    for vv,cc in vectors:
        ax_vectors.quiver(0,0,0,vv[0][0],vv[0][1],vv[0][2] , color = cc,  length = 2)
    
    vel_aux = vel_init
    vel_init = vel_after
    rot = 0
    
    plt.pause(0.1)