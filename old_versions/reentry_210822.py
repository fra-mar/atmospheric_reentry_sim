#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:49:37 2021

@author: paco
"""

'''
Information for map and start points:
Start in 38.9N,50.40E at 62Km atitude. Map from this point to 1700 km away 
from primary landing point at 47.10,69.30 decimal coordinates
'''
import numpy as np

from matplotlib import pyplot as plt

from mpl_toolkits import mplot3d

from pynput import keyboard

from lift import  lift_vector, air_density, gravity_calc, area_drag, landing_sites

#from lin_transf_matrix_1 import lift_function_beta, find_angle

#%% Axes definiton

plt.ion()

fig = plt.figure('Soyuz, reentry path', figsize = (16,5))

ax = fig.add_gridspec(3,5)

plt.subplots_adjust(wspace = 0.8)

ax_xy = fig.add_subplot(ax[0:,2:]) #ax for surface, bird view

img = 'Kazakhstan_map.png'

map = plt.imread(img)

extents = (46.000,88.000,40.000,56.000)

ax_xy.imshow(map, aspect = 'auto') 

#ax_h = fig.add_subplot(1,2,1)

ax_vectors =  fig.add_subplot(ax[:,0:2],projection = '3d')

ax_vectors.view_init(20,-30)

x_ax_vectors = np.linspace(-7,7,7)
y_ax_vectors = np.linspace(-7,7,7)
z_ax_vectors = y_ax_vectors.copy()

X,Y = np.meshgrid(x_ax_vectors,y_ax_vectors)

Z=0*X + 0*Y

YY,ZZ = np.meshgrid ( y_ax_vectors,z_ax_vectors)

XX = 0*Y + 0*Z

ax_vectors.invert_xaxis()



#%% Lambdas and coversions

to_rad = lambda dg: dg*np.pi/180   #easy formula to express degreen in rads

to_degrees = lambda rad: rad* 180 / np.pi

module = lambda v: np.sqrt( np.sum(v**2)) 

antivector_u = lambda v: -1 * v/module(v)

v_unitary = lambda v: v/module(v)

k = 111316.5


#%% initial variables

A = 3.8 ; m = 2300.0; rot = 1 #100Kg mass #area of the body D=2.2m ;  rot = flying upward (0 upside-down)

Cd = 1.26  #drag coefficient, somehow according to 
#https://www.faa.gov/about/office_org/headquarters_offices/avs/offices/aam/cami/
#library/online_libraries/aerospace_medicine/tutorial/media/iii.4.1.7_returning_from_space.pdf

A_lift = 3.8  #area of the surface intended as wing

Cl = 0.5  #lift coefficient
    
vel_init = np.array([[4720.7231,5956.0702,-165.01]]) # #for an orbit inclination 51.6 degrees

vel_after = np.array([[4720.7231,5956.0702,-179.01]]) #for a reentry angle aprox 1.35 degrees

vel_aux = vel_init #this will help to prevent runtime error when calculating lift vector

vel_original = vel_init #reference vector to calculate linear transformation matrix

v_lift_original = np.cross(vel_after,   np.cross(vel_original,vel_after)    )

theta = 0 #angle in degrees. angle intended to turn the lift vector about velocity axis

x,y = -6e4,0 # x NS axis, y WE axis...in a 3D aixs

h = 1e3*70; graph_lim = h #meters #h similar to z ina 3D axis

rho = air_density(h) #air density 

lat_init = 40.00; long_init =54.00

lat, long = lat_init, long_init

total_time = 0

status = np.array ([0., y,x, h, 0.]) #time, hor_position, vert_position, altitude, g force
 
#%% start vectors and angles
x_u= np.array( [[1,0.0,0.0]])

y_u = np.array( [[0.,1.,0.]])

z_u = np.array( [[0.,0.,1]])

alpha, beta, gamma = 0,0,0


#%% Controlls

def on_press(key):
    
    global rot
    if key.char == 'z':
        rot = 1 
    if key.char == 'x':
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
    #ax_data.clear()
    
    
    if rot == 1 or rot == -1:
        print ('yep')
        if rot == 1:
            theta += np.deg2rad(5)
        elif rot == -1:
            theta -=np.deg2rad(5)
        
        print (rot,np.rad2deg(theta) )
    
    t =  0.1 #second
    
    rho = air_density(h)
    
    A, par_status = area_drag(h)
    
    v_mod = module (vel_init)
    
    D_mod = 0.5 * (v_mod**2)*A*Cd*rho   #calculates drag
    
    D = antivector_u(vel_init) * D_mod         
    
    L_mod = 0.5 * (v_mod**2)*A_lift*Cl*rho    #calculates lift
    
    L_mod = D_mod * 0.4
    
    v_lift_u, v_lift_r_u = lift_vector (v_unitary(vel_after), theta)
    
    L = v_lift_r_u * L_mod 
    
    G = gravity_calc(h,m)                  #calculates gravity as function of h
    
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
    
    long = long + y_displ / k 
    
    g_s =( module(R)/m ) / 9.8
    
    status = np.vstack ((status, [total_time,y,x, h, g_s]))
    
    total_time += t
    
    heading = np.rad2deg( np.arctan2 (y_displ,x_displ) )
    if heading<0:
        heading = 360 + heading
    
    long_prim, lat_prim = 69.30, 47.10
    
    dif_long, dif_lat = long_prim-long, lat_prim-lat
    
    
    prim_bearing = np.rad2deg( np.arctan2 (dif_long, dif_lat) )
    if prim_bearing<0:
        prim_bearing = 360 + heading
    
    print (L_mod/D_mod)
    
    ax_xy.set_title ('Primary landing site (star). Secondaries (triangles)')
    
    y_ticks = [y+40 for y in range (3,16,3)]
    x_ticks = [x+45 for x in range (3,42,3)]
    ax_xy.set_xticks(x_ticks); ax_xy.set_yticks(y_ticks)
    
    
    #ax_xy.scatter(y, x, color = 'b') 
    ax_xy.scatter(long,lat,color = 'b')
    ax_xy.imshow(map,extent = extents,aspect  = 'auto', alpha = 0.5)
    ax_xy.scatter(69.30, 47.10, marker = '*', color = 'red') #primary landing point
    for l in range(1, len(landing_sites)):
        c = landing_sites.get(str(l))
        ax_xy.scatter (c[0], c[1], marker = 'v', color = 'gray')
        
    
    #Data display
    altitude_label = 'Altitude {:.2f} m'.format(h)
    ground_speed = 'GS {:.2f} m/s'.format ( module ( np.array([vel_after[0][0],vel_after[0][1]])))
    vertical_speed = 'VS {:.2f} m/s'.format (vel_after[0][2])
    density_str = 'Rho {:.4} Kg/m3'. format (rho)
    parachute_str = 'Parachute '+ par_status
    g_string = 'G forces: {:.2f}'.format(g_s)
    cap_rot_str = 'CapRot: {:3.3} deg'. format(np.rad2deg(theta))
    lat_str = 'Lat: {:.3f}'.format(lat)
    long_str = 'Long: {:.3f}'. format(long)
    heading_str = 'Heading: {:.2f} deg'.format(heading)
    prim_bearing_str = 'LSiteHeading: {:.2f} deg'.format(prim_bearing)
    ax_xy.text(80,55, altitude_label)
    ax_xy.text(80,54, vertical_speed)
    ax_xy.text(80,53, ground_speed)
    ax_xy.text(80,52, g_string)
    ax_xy.text(80,51, cap_rot_str)
    ax_xy.text(80,50, parachute_str)
    ax_xy.text(80,49, lat_str)
    ax_xy.text(80,48, long_str)
    ax_xy.text(80,47, heading_str)
    ax_xy.text(80,46, prim_bearing_str)
    ax_xy.grid()
    
    #vectors plot.
    
    ax_vectors.text(0, -9, -8, 'W',)
    ax_vectors.text(0, 9, -8, 'E',)
    ax_vectors.text(-9, 0, -8, 'S',)
    ax_vectors.text(9, 0, -8, 'N',)
    surf1 = ax_vectors.plot_surface(X, Y, Z, alpha = 0.01) #planes for better visual vectors
    surf2 = ax_vectors.plot_surface (XX,YY,ZZ, alpha = 0.01)
    ax_vectors.set_xticklabels([])
    ax_vectors.set_yticklabels([])
    ax_vectors.set_zticklabels([])
    '''ax_vectors.set_xlabel('x')
    ax_vectors.set_ylabel('y')
    ax_vectors.set_zlabel('z')'''
    ax_vectors.invert_xaxis()
    ax_vectors.set_title ('Velocity: red. Lift: blue. Drag: orange. Resultant: green. G: black')
    
    R_u, G_u, D_u = v_unitary(R), v_unitary(G),  v_unitary(D)
    vel_after_u = v_unitary(vel_after)
    vectors = [ (vel_after_u,'red'),
               (G_u, 'black'),(R_u, 'green'), (v_lift_r_u, 'blue')]
    v_lift_r = v_lift_r_u * L_mod
    vectors_beta = [ (vel_after*5,'red'),
               (G, 'black'),(R, 'green'), 
               (D, 'orange'),(v_lift_r, 'blue')] 
    for vv,cc in vectors_beta:
        ax_vectors.quiver(0,0,0,vv[0][0],vv[0][1],vv[0][2], length = 0.00015, color = cc)
    
    vel_aux = vel_init
    vel_init = vel_after
    rot = 0
    
    plt.pause(t)

#%% Plotting flight analysis

fig2 = plt.figure('flight analysis')
ax2 = fig2.add_gridspec(2,2)

ax2_h = fig2.add_subplot(ax2[0,0])
ax2_h.scatter(status[:,0], status[:,3])

ax2_yx = fig2.add_subplot(ax2[0,1])
ax2_yx.scatter(status[:,1], status[:,2])

ax2_g_s = fig2.add_subplot( ax2[1,0])
ax2_g_s.scatter(status[:,0], status[:,4] )

plt.show()
#ax_xy = fig.add_subplot(ax[0:,2:]) #ax for surface, bird view
