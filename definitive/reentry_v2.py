#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Please read README.txt in...

Reentry simulates the path of a Soyuz capsule through the atmosphere on its 
way back to Earth. 
My goals were to keep it simple, and learn myself more about physics and programming.
Thatś why only numpy, matplotlib and pynput libraries are used.

The Soyuz or Apollo reentry modules has an offset center of mass that makes it tilt. 
This assymmetric configuration generates som lift that can be used to change her direction: 
during the simulation start the vector is pointing up (0deg).
If the ship is rotated the vector points to the sides, changing the flight 
direction. 

Letters z (counterclockwise, positve degrees) and x (clockwise,
negative degrees) are used to rotate the capsule. Easy

Some parameters are stored in the (numpy) array flight_data that can be saved at the 
end of a simulation for further analysis.

The simulation can be interrupted by pressing q.

Regarding initial drag coefficient and lift to drag ratio I recommend default
for 'realism' or Cd=0.5 L/D=1.5 for longer, more controllable flights.

Cd changes with a variety of factors. For simplicity, I assume a constant angle 
of attack, and a increased Cd with lower air density, as a very simplified 
surrogate of Reynolds number.

In the bottom right corner vectors are displayed in a side view, perpendicular to 
the direction of flight.
Vectors are drawn as well along with the ship projected on the horizontal plane
The velocity vector has its own scale. Forces have the same scale.

Flight data are stored in a np.array ('flight_data.npy') that can be visualized 
with the script flight_analysis211108.py

Enjoy

'''
import numpy as np

from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation

import matplotlib.patches as patches

from mpl_toolkits.mplot3d import axes3d

from pynput import keyboard

from lift import  lift_vector, air_density, gravity_calc, area_drag, landing_sites

import os

pwd= os.getcwd()
with open (os.path.join(pwd, 'lib_files', 'init_params.txt')) as f:
    line= f.readline()
    mass, LD_ratio= line.split(',')
mass= float(mass)
LD_ratio= float(LD_ratio)

#%% Ask user what Cd and L/D ratio

Cd = 1 

#Draws equation to correct Cd after rho.
rho_min, rho_max = air_density(1e4), air_density(7e4)

Cd_high = Cd *1.4 #at higher atmosphere Cd will be 40% higher compared to lower atm. 

# asuming equation in the form y = Ax^n, i.e., Cd = k * rho^k ...
n = (np.log10(Cd) - np.log10(Cd_high)) / (np.log10(rho_min) - np.log10(rho_max) )

log_A = n * (-np.log10(rho_max)) + np.log10(Cd_high)

#%% Lambdas and coversions

module = lambda v: np.sqrt( np.sum(v**2)) 

antivector_u = lambda v: -1 * v/module(v)

v_unitary = lambda v: v/module(v)

ortovector_u = lambda v: np.array([ -v[1],v[0]]) /module (v)

k = 111316.5 #meters/deg at the equator

#%% Keyboard ontrolls

def on_press(key):
    
    global rot
    if key.char == 'z':
        rot = 1 
    if key.char == 'x':
        rot = -1
    else:
        pass

listener=keyboard.Listener(on_press=on_press)
listener.start()  

#%% initial variables

A = 3.8 ; m = mass; rot = 1 #projected area Soyuz, mass = 2.3 tons

h = 70.1e3 #altitud in meters

rho = air_density(h) #air density in Kg/m3

vel_init = np.array([[4720.7231,5956.0702,-1.01]]) # for orbit incl 51.6deg

vel_after = np.array([[4720.7231,5956.0702,-2.01]]) 

theta = 0 #angle in degrees. angle intended to turn the lift vector about velocity axis

lat_init = 40.0; long_init = np.random.randint(46,75)

lat, long = lat_init, long_init

total_time = 0

#Start vectors_list
G = np.array([[0., 0., 0.]])
L = np.array([[0., 0., 0.]])
D = np.array([[0., 0., 0.]])
R = np.array([[0., 0., 0.]])

vectors_list = [(vel_after,1e3,'red','Velocity'), (G,1e5,'black','Gravity'),
                (L,1e5,'blue','Lift'), (D,1e5,'orange','Drag'),
                (R,1e5,'green','Resultant')]


#flight data: time, long, lat, altitude, g force, rho, vel, Cd, lift, drag, gravity, resultant
flight_data = np.array ([0., long,lat, h, 0,0,module(vel_after),Cd,0,0,0,0]) 


#%% Function to stop animation under a given altitude
def gen_frames(): #generates a number of frames if a condition is fullfilled
    global h
    frame = 0
    while h > 8e3: #under 8000m nothing happens, that's why sim stops here
        frame += 1
        yield frame
#%% Main data generation

def gen_data():  
    
    global long, lat, h, theta, vel_after, vel_init, total_time
    global rot, G, L, D, R, flight_data
       
    if rot == 1 or rot == -1: #change Soyuz's rotation angle
        
        if rot == 1:
            theta += np.deg2rad(2.3)
            if theta>np.pi: #change sig if over 180 deg
                theta = -np.pi - (theta-np.pi)
            rot = 0
        elif rot == -1:
            theta -=np.deg2rad(2.4)
            if theta<-np.pi:
                theta = np.pi - (theta-(-np.pi))
            rot = 0
    t =  0.1 #real seconds. But simulation goes slower...unfortunately
    
    rho = air_density(h)
    
    A, par_status = area_drag(h) #projected area for drag calculation. Increased with parachute deploying
    
    v_mod = module (vel_init) #module of vector velocity for drag force calculation
    
    Cd = (10**log_A) * rho**n #simplified calculation of Cd varying with rho
    
    D_mod = 0.5 * (v_mod**2)*A*Cd*rho   #calculates drag
    
    D = antivector_u(vel_init) * D_mod         
    
    L_mod = D_mod * LD_ratio #calculates lift after L/D ratio
    
    #function lift_vector gets the unitary, normal (vertical) to velocity vector and
    #rotates that vector theta radians around velocity vector.
    
    v_lift_r_u = lift_vector (v_unitary(vel_after), theta)
    
    L = v_lift_r_u * L_mod #calculates lift as a module * unitary_vector
    
    G = gravity_calc(h,m)  #calculates gravity as function of h
    
    R = G + D + L          #resultant force
    
    a = R / m              #acceleration vector 
        
    vel_after = vel_init  + a * t                      #vectorial calculation
    
    h = h + vel_after[0][2]*t + 0.5*a[0][2]*t**2   #kinematic eq for x,y,z axis
    
    x_displ = vel_after[0][0]*t + 0.5*a[0][0]*t**2 
    
    y_displ = vel_after[0][1]*t + 0.5*a[0][1]*t**2
    
    #longitud change straigth forward with k (meters/degree)
    #latitude more complicated due to change in k with latitude...
    #...shorter distance to Earth axis with higher/lower latitudes
    lat = lat + x_displ/(np.cos( np.deg2rad(lat) ) * k )
    
    long = long + y_displ / k 
    
    g_s = module(a) / 9.8  #g forces
    
    #Calculates heading
    heading = np.rad2deg( np.arctan2 (y_displ,x_displ) )
    if heading<0:
        heading = 360 + heading
        
    total_time += t
    vel_init = vel_after
    
    flight_data = np.vstack ((flight_data,
                              [total_time,long,lat, h, g_s,rho,v_mod,Cd,
                              L_mod,D_mod,module(G),module(R)]))
    np.save('flight_data.npy', flight_data)
    np.savetxt('flight_data.csv', flight_data, delimiter=',', 
               header = 'Time,Long,Lat,Alt,g_s,rho,vel,Cd,Lift,Drag,Gravity,Resultant')
    
    return (h, vel_after, g_s, theta, lat, long, heading,
            par_status, total_time, D,G,L,R)

#%% Starting plots

fig = plt.figure('Soyuz, reentry path', figsize = (16,10))

ax = fig.add_gridspec(3,5)

plt.subplots_adjust(wspace = 0.8)

ax_xy = fig.add_subplot(1,1,1) #axis for map view

ax_xy.set_title ('Primary landing site (star). Secondaries (triangles)')
ax_xy.set_xlabel ('Longitude', fontsize = 10)
ax_xy.set_ylabel ('Latitude', fontsize = 10)

extents = (46.000,88.000,40.000,56.000)

ax_xy.set_xlim([extents[0],extents[1]])
ax_xy.set_ylim([extents[2],extents[3]])

img = 'Kazakhstan_map.png'

map = plt.imread(img)

ax_xy.imshow(map,extent = extents,aspect  = 'auto', alpha = 0.5)

y_ticks = [y+40 for y in range (3,16,3)]
x_ticks = [x+45 for x in range (3,42,3)]
ax_xy.set_xticks(x_ticks)
ax_xy.set_yticks(y_ticks)
ax_xy.grid()

#pins primary and secondary landing sites
ax_xy.scatter(69.30, 47.10, marker = '*', color = 'red') 
for l in range(1, len(landing_sites)):
    c = landing_sites.get(str(l))
    ax_xy.scatter (c[0], c[1], marker = 'v', color = 'gray')

#displayed data labels
ax_xy.text(69,55, 'Altitude', fontsize = 12) 
ax_xy.text(80,55, 'GroundSpeed', fontsize = 12)
ax_xy.text(80,52.5, 'VerticalSpeed', fontsize = 12)
ax_xy.text(80,50, 'G forces', fontsize = 12)
ax_xy.text(80,48, 'CapsuleRotation', fontsize = 12)
ax_xy.text(83,50, 'Parachute', fontsize = 12)
ax_xy.text(47,55, 'Latitude', fontsize = 12)
ax_xy.text(54,55, 'Longitude', fontsize=12)
ax_xy.text(61,55, 'Heading', fontsize = 12)


#text indicators, initialize
h_txt= ax_xy.text(72,55, '', fontsize = 20)
ground_speed_txt = ax_xy.text(80,54.2,'', fontsize = 20)
vertical_speed_txt= ax_xy.text(80,51.7,'', fontsize = 20)
g_txt= ax_xy.text(80,49.3, '', fontsize = 20)
cap_rot_txt= ax_xy.text(80,47.1, '', fontsize = 20)
par_status_txt= ax_xy.text(83,49.3, '', fontsize = 20)
lat_txt= ax_xy.text(49.0,55, '', fontsize = 20)
long_txt= ax_xy.text(57,55, '', fontsize=20)
heading_txt= ax_xy.text(64.5,55, '', fontsize = 20)

#starting ship
ship, = ax_xy.plot([],[], 'bo')

#drawing a rectangle for sideview
ax_xy.add_patch( patches.Rectangle(xy = (80,40.8), width = 7.8, height = 5,
                                   linewidth = 1,
                                   color = 'white', 
                                   ec = '#F58D8D',
                                   alpha = 0.5) )
ax_xy.text(85.5,45.5, 'Side view', fontsize = 8)

#starting vector plots
ax_xy.scatter (82,43, s = 50, c = 'blue')

q_vel_after_z = ax_xy.quiver(82, 43, 0, 0,
                   width= 2e-3,scale = 7e4, color = 'red', label = 'Velocity') 
q_vel_after_yx = ax_xy.quiver(0,0, 0, 0,
                   width= 1e-3,scale = 7e4, color = 'red')

q_D_z = ax_xy.quiver(82, 43, 0, 0,
                   width= 2e-3,scale = 6e5, color = 'orange', label = 'Drag')
q_D_yx = ax_xy.quiver(0, 0, 0, 0,
                   width= 1e-3,scale = 6e5, color = 'orange')

q_L_z = ax_xy.quiver(82, 43, 0, 0,
                   width= 2e-3,scale = 6e5, color = 'blue', label = 'Lift')
q_L_yx = ax_xy.quiver(0,0, 0, 0,
                   width= 1e-3,scale = 6e5, color = 'blue')

q_G_z = ax_xy.quiver(82, 43, 0, 0,
                   width= 2e-3,scale = 6e5, color = 'black', label = 'G')

q_R_z = ax_xy.quiver(82, 43, 0, 0,
                   width= 2e-3,scale = 6e5, color = 'green', label = 'Resultant')
q_R_yx = ax_xy.quiver(0, 0, 0, 0,
                   width= 1e-3,scale = 6e5, color = 'green')

ax_xy.legend(loc = 4, ncol = 5)


#%% Animation

def update(frame):  
    
    #global status, status_DL
    global q_D_z, q_vel_after_z, q_G_z, q_L_z, q_R_z
    global q_vel_after_yx, q_D_yx, q_L_yx, q_R_yx
    global flight_data
    (h, vel_after, g_s, theta, lat, long, heading, par_status,
    total_time, D, G, L, R) = gen_data()
    
    ship.set_data(long, lat)
    
    vel_after_yx = np.array([vel_after[0][1], vel_after[0][0]])
    vel_after_yxz = np.array( [ module(vel_after_yx), vel_after[0][2]])
    q_vel_after_z.set_UVC(vel_after_yxz[0], vel_after_yxz[1])
    q_vel_after_yx.set_UVC(vel_after_yx[0], vel_after_yx[1])
    q_vel_after_yx.set_offsets(np.array( [long,lat]))
    
    Dyx = np.array([D[0][1], D[0][0] ] )
    Dyxz = np.array( [ module(Dyx)* -1 ,D[0][2]]) #DragVector displayed side view
    q_D_z.set_UVC(Dyxz[0], Dyxz[1])
    q_D_yx.set_UVC(Dyx[0], Dyx[1])
    q_D_yx.set_offsets(np.array( [long,lat]))
    
    #projection of L in side view
    orto_u_vel_after = ortovector_u(vel_after_yxz)
    lift_z = L[0][2]
    alpha = np.arctan(orto_u_vel_after[0]/orto_u_vel_after[1])
    mod_Lyxz = lift_z/np.cos(alpha)
    Lyxz = orto_u_vel_after*mod_Lyxz
    
    q_L_z.set_UVC(Lyxz[0], Lyxz[1])
    Lyx = np.array([L[0][1], L[0][0] ] )
    q_L_yx.set_UVC(Lyx[0], Lyx[1])
    q_L_yx.set_offsets(np.array( [long,lat]))

    Gyxz = np.array( [ 0., G[0][2]])
    q_G_z.set_UVC(Gyxz[0], Gyxz[1])
    
    Ryxz = Dyxz + Lyxz + Gyxz
    q_R_z.set_UVC(Ryxz[0], Ryxz[1])
    Ryx = Dyx + Lyx 
    q_R_yx.set_UVC(Ryx[0], Ryx[1])
    q_R_yx.set_offsets(np.array( [long,lat]))

    #Displayed data: updating content
    ground_speed = '{:.2f} m/s'.format ( module ( np.array([vel_after[0][0],vel_after[0][1]])))
    vertical_speed = '{:.2f} m/s'.format (vel_after[0][2])
    g_string = '{:.2f}'.format(g_s)
    cap_rot_str = '{:3.2f}'. format(np.rad2deg(theta))
    lat_str = '{:.2f}'.format(lat)
    long_str = '{:.2f}'. format(long)
    heading_str = '{:.2f}'.format(heading)
    
    #Displayed data: display content
    h_txt.set_text('{:.2f} m'.format(h))
    ground_speed_txt.set_text(ground_speed)
    vertical_speed_txt.set_text(vertical_speed)
    g_txt.set_text(g_string)
    cap_rot_txt.set_text(cap_rot_str)
    par_status_txt.set_text(par_status)
    lat_txt.set_text(lat_str)
    long_txt.set_text(long_str)
    heading_txt.set_text(heading_str)
    
    
    return (ship, q_D_z, q_vel_after_z, q_G_z, q_L_z, q_R_z,
            q_vel_after_yx, q_D_yx, q_L_yx, q_R_yx,
            h_txt, ground_speed_txt, vertical_speed_txt, g_txt,
            cap_rot_txt, par_status_txt, lat_txt, long_txt, heading_txt,
           )
   
ani = FuncAnimation(fig, update, frames = gen_frames, interval = 20,
                    repeat = False, blit=True )
plt.show()
