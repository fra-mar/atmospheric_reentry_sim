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

Start in 38.9N,50.40E at 62Km atitude. Map from this point to 1700 km away from primary landing point at 47.10,69.30 decimal coordinates
'''
import numpy as np

from matplotlib import pyplot as plt

from mpl_toolkits import mplot3d

from pynput import keyboard

from lift import  lift_vector, air_density

#from lin_transf_matrix_1 import lift_function_beta, find_angle


#%% Lambdas and coversions

to_rad = lambda dg: dg*np.pi/180   #easy formula to express degreen in rads

to_degrees = lambda rad: rad* 180 / np.pi

module = lambda v: np.sqrt( np.sum(v**2)) 

antivector_u = lambda v: -1 * v/module(v)

v_unitary = lambda v: v/module(v)

k = 111316.5  #1 degreee equivalence to meters at equator#############################SIGUE AQUI!!!!!




#%% initial variables

A = 2.2 ; m = 2300.0; rot = 1 #100Kg mass #area of the body 1 m2 rot = flying upward (0 upside-down)

Cd = 1  #drag coefficient

A_lift = 0.8  #area of the surface intended as wing

Cl = 2   #lift coefficient
    
vel_init = np.array([[5600.,4500,-165.01]]) # #for an orbit inclination 51.6 degrees

vel_after = np.array([[5600.,4500,-179.01]]) #for a reentry angle aprox 1.35 degrees

vel_aux = vel_init #this will help to prevent runtime error when calculating lift vector

vel_original = vel_init #reference vector to calculate linear transformation matrix

v_lift_original = np.cross(vel_after,   np.cross(vel_original,vel_after)    )

theta = 0 #angle in degrees. angle intended to turn the lift vector about velocity axis

x,y = 0,0 # x NS axis, y WE axis...in a 3D aixs

h = 1e3*60; graph_lim = h #meters #h similar to z ina 3D axis

lat_init = 40.00; long_init =48.00

lat, long = lat_init, long_init

rho = air_density(h) #air density 

total_time = 0

status = np.array ([y,x, h])
status_forces = np.array ( [0,0,0] )
status_angle = np.array ( [0])

rot, act = 0, 0
#%% other functions for other parameters
def gravity_calc (h):
    global m
    grav_constant = 6.67408*1e-11 #units m3 kg-1 s-2
    radius_earth = 6.371*1e6   #units in m
    earth_mass = 5.9722*1e24 #units in kg
    Gy = grav_constant * earth_mass * m / ( (radius_earth + h)**2 )
    G = np.array( [[0,0,-Gy]])
    return G

def area_drag(h): # simulates parachutes deploying under 10.7km altitude
    #real soyuz deploys brake parachute at 10.7Km and main at 8.5Km. 
    #here parachutes deployed lower altitud for more versatility
    if h>= 4e3 :
        A_drag = 2.2
        par_status = 'packed'
    elif h<4e3 and h >=  3e3:
        A_drag = 24
        par_status = 'brake'
    elif h <3e3:
        A_drag = 1e3
        par_status = 'main'
    return A_drag, par_status
    
#%% start vectors and angles
x_u= np.array( [[1,0.0,0.0]])

y_u = np.array( [[0.,1.,0.]])

z_u = np.array( [[0.,0.,1]])

alpha, beta, gamma = 0,0,0

#%% Axes definiton

plt.ion()

fig = plt.figure('fall analysis', figsize = (16,6))

ax = fig.add_gridspec(6,6)

ax_map = fig.add_subplot ( ax[1:,3:])

#img = 'Kazakhstan_map.png'
#map = plt.imread(img)
#extents = (46.000,88.000,40.000,56.000)

#ax_map.imshow(map, aspect = 'auto', extent = extents )   #this sets the coordinates for the map (image)

#ax_xy = fig.add_subplot(ax[0:,1:]) #ax for surface, bird view
#ax_xy = fig.add_subplot(ax[4:,2]) 

#ax_h = fig.add_subplot(1,2,1)

#ax_vectors =  fig.add_subplot(ax[0:2,0:2],projection = '3d')

#ax_vectors.view_init(20,30)

ax_data = fig.add_subplot(ax[2:,0:2])


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
    #ax_xy.clear()
    #ax_vectors.clear()
    ax_data.clear()
    
    if rot == 1 or rot == -1:
        print ('yep')
        if rot == 1:
            theta += np.deg2rad(10)
        elif rot == -1:
            theta -=np.deg2rad(10)
        
        print (rot,np.rad2deg(theta) )
    
    t =  0.1 #second
    
    rho = air_density(h)
    
    A, par_status = area_drag(h)
    
    print ('rho: {:.2f} kg/m3'.format(rho))
    
    v_mod = module (vel_init)
    
    D_mod = 0.5 * (v_mod**2)*A*Cd*rho   #calculates drag
    
    D = antivector_u(vel_init) * D_mod         
    
    L_mod = 0.5 * (v_mod**2)*A_lift*Cl*rho    #calculates lift
    
    v_lift_u, v_lift_r_u = lift_vector (v_unitary(vel_after), theta)
    
    L = v_lift_r_u * L_mod 
    
    G = gravity_calc(h)                  #calculates gravity as function of h
    
    #L = np.array( [[0.,0.,0.]])
    
    R = G + D + L                        #resultant force
    
    a = R / m   
        
    vel_after = vel_init  + a * t                      #vectorial calculation
    
    h = h + vel_after[0][2]*t + 0.5*a[0][2]*t**2  
    
    x_displ = vel_after[0][0]*t + 0.5*a[0][0]*t**2 
    
    y_displ = vel_after[0][1]*t + 0.5*a[0][1]*t**2
    
    x = x + x_displ
    
    y = y + y_displ
    
    lat = lat + x_displ/(np.cos( np.deg2rad(lat) ) * k )
    
    long = long + y_displ / k     #############################33SIGUE AQUI!!!!!!!!!!!!
    
    g_s =( module(R)/m ) / 9.8
    
    status = np.vstack ((status, [y,x, h]))
    
    status_forces = np.vstack( (status_forces, [D_mod, L_mod, g_s]) )
    
    #status_angle = np.vstack ( (status_angle, [np.rad2deg( find_angle ( L, vel_after))]) )
    
    total_time += t
    
    #ax_xy.set_title ('West to East, North/South')
    
    #ax_xy.set_xlim([-100,9*1e5]); ax_xy.set_ylim([-graph_lim,graph_lim])
    
    #ax_xy.scatter(y, x, color = 'b')
    #ax_xy.scatter(long,lat, color = 'b')
    
    #extents = (46.000,88.000,40.000,56.000)
    ax_map.set_xlim([46.000,88.000]); ax_map.set_ylim([40.000,56.000])
    ax_map.scatter(long,lat, color = 'b')
    
  
    
        
    
    altitude_label = 'Altitude {:.2f} m'.format(h)
    ground_speed = 'GS {:.2f} m/s'.format ( module ( np.array([vel_after[0][0],vel_after[0][1]])))
    vertical_speed = 'VS {:.2f} m/s'.format (vel_after[0][2])
    density_str = 'Rho {:.4} Kg/m3'. format (rho)
    parachute_str = 'Parachute '+ par_status
    g_string = 'G forces: {:.2f}'.format(g_s)
    cap_rot_str = 'CapRot: '+str(theta)
    ax_data.text(.01,.8, altitude_label)
    ax_data.text(.01,.6, vertical_speed)
    ax_data.text(.01,.4, ground_speed)
    ax_data.text(.01,.2, g_string)
    ax_data.text(.5,.4, density_str)
    ax_data.text(.5,.2, parachute_str)
    ax_data.text(.5,.6, cap_rot_str)
    
    
    #ax_xy.grid()
    #ax_vectors.scatter([-1,1,0,0,0,0],[0,0,-1,1,0,0],[0,0,0,0,1,-1])
    #ax_vectors.set_xlabel('x')
    #ax_vectors.set_ylabel('y')
    #ax_vectors.set_zlabel('z')
    
    R_u, G_u, D_u = v_unitary(R), v_unitary(G),  v_unitary(D)
    vel_after_u = v_unitary(vel_after)
    vectors = [ (vel_after_u,'red'),
               (G_u, 'black'), (v_lift_r_u, 'blue')]
    #for vv,cc in vectors:
        #ax_vectors.quiver(0,0,0,vv[0][0],vv[0][1],vv[0][2] , color = cc,  length = 2)
    
    vel_aux = vel_init
    vel_init = vel_after
    rot = 0
    
    plt.pause(0.1)