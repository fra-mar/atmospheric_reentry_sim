#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:49:37 2021
Updated Tue Nov 01, 2022

This script simulates a falling object, asymmetric so it crates some kind
of Lift/Drag ratio. By rotating the object 180deg it becomes 
slightly controllable.
By changing the L_D_ratio it becomes even more controllable.

Reasonable parameters to simulate a space capsule are 
h=6e4,v_x= 7e3, m= 2.5e3, L_D_ratio= 0.3, A= 2.2
OBS! (air density fixed to 1.225!!!)

For a wingsuiter h=4e3, v_x=4, L_D_ratio= 1.2, m= 80, A= 0.5

Updates:
Updatable objects (ship, vectors, time text) wasn't updated.
Fixed with setting blit = False and clearing collections
( ax_h.collections.clear())
OBS! 230629 I commented that line and set blit=True..cause .clear raised error
...I guess related to matplotlib version.


"""
import numpy as np

from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation

from pynput import keyboard



#%% Lambdas, calculators and coversions

to_rad = lambda dg: dg*np.pi/180   #easy formula to express degreen in rads

to_degrees = lambda rad: rad* 180 / np.pi

module = lambda v: (v[0]**2 + v[1]**2)**(1/2)

angle = lambda v: np.arctan(v[1]/v[0])

antivector_u = lambda v: -1 * v/module(v)

ortovector_u = lambda v: np.array([ -v[1],v[0]]) /module (v)

def gravity_calc (h, m):
    grav_constant = 6.67408*1e-11 #units m3 kg-1 s-2
    radius_earth = 6.371*1e6   #units in m
    earth_mass = 5.9722*1e24 #units in kg
    Gy = grav_constant * earth_mass * m / ( (radius_earth + h)**2 )
    G = np.array( [0,-Gy])
    return G

#%% Controlls
def on_press(key):
    
    global rot
    if key.char == 'x':
        rot = 1
    elif key.char == 'z':
        rot = -1
    else:
        pass
    #print (str(rot),'\n')

listener=keyboard.Listener(on_press=on_press)
listener.start()

#%%Time management

interval = 50 #interval between frames in MILISECONDS
t =  0.1 # interval between calculations (of v, h, s...)
time_acceleration = t/(interval*1e-3)

#%% initial variables

rho = 1.225 #air density

A = 2.2 #area facing direction of movement m2
m = 2.2e3 #mass in Kg
L_D_ratio = 0.3 #lift/drag coefficient
h = 1e4 #m
v_x = 300 #m/s horizontal


rot = 1 #rot = flying upward (0 upside-down)

Cd = 1  #drag coefficient

v = np.array([v_x,0.1]) # vx 2 m/s backwards and 4 m/s upwards

v_mod = module (v)

G = np.array([0,-9.8])*m # G-FORCE

D_mod = 0.5 * (v_mod**2)*A*Cd*rho

D = antivector_u (v) * D_mod

L_mod = L_D_ratio * D_mod  # for a L/D ratio 0.3

L = ortovector_u (v) * L_mod * rot

R = G + D + L

a = R/m

s = 0

graph_lim = h #meters

total_time = 0

vectors_list = [(v,1e3,'red','Velocity'), (G,1e5,'black','Gravity'),
                (L,1e5,'blue','Lift'), (D,1e5,'orange','Drag'),
                (R,1e5,'green','Resultant')]

status = np.array ([total_time, v[0],v[1],0,0, h])

status_DL = np.array( [ total_time,D[0],D[1],L[0],L[1]] )

#%% Main loop

def gen_data():

    global t, s, h, v, v_mod,  total_time, L_D_ratio
    global status, status_DL,vectors_list
    
    D_mod = 0.5 * (v_mod**2)*A*Cd*rho   #calculates drag
    
    D = antivector_u(v) * D_mod

    L_mod = L_D_ratio * D_mod   #calculates lift

    L = ortovector_u (v) * L_mod * rot
    
    G = gravity_calc(h, m)                  #calculates gravity as function of h
    
    R = G + D + L                        #resultant force
    
    a = R / m   
        
    v = v  + a * t                      #vectorial calculation
    
    v_mod = module (v)
    
    h = h + v[1]*t + 0.5*a[1]*t**2
    
    if h < 5:   # when the ship touches the ground
        v = np.array ([0.01,0.01])
        h = 5
        
    
    s = s + v[0]*t + 0.5*a[0]*t**2
    
    #update arrays for later analysis
    status_new = np.array ([total_time,v[0],v[1],L[0],L[1], h])
    status_DL_new = np.array([ total_time,D[0],D[1],L[0],L[1]])
    status = np.vstack ((status,status_new))
    status_DL = np.vstack( (status_DL, status_DL_new))
    
    #vectors_to_plot = (v,G,L,D,G)
    
    vectors_list = [(v,1e3,'red','Velocity'), (G,1e5,'black','Gravity'),
                    (L,1e5,'blue','Lift'), (D,1e5,'orange','Drag'),
                    (R,1e5,'green','Resultant')]
    
    np.save('status.npy', status)
    np.save('status_DL.npy', status_DL)
    
    total_time += t
   
    
    return s, h, vectors_list

#%% Starting plots

fig = plt.figure('Asimmetric spherical object. Free fall simulation', figsize =
                 (14,8), facecolor= '#ffe4c4')

ax_h = fig.add_subplot(1,1,1)

ax_h.set_title ('Mass: {:} Kg.  V: {}m/s horizontal.  L/D ratio: {:.2f}'.
                format(m, v_x, L_D_ratio), fontsize = 18,
                color= 'C5')
ax_h.set_ylabel ('Altitude (m)', fontsize= 16)
ax_h.set_xlabel ('Ground distance from start (m)', fontsize= 16)
ax_h.set_xlim([-50,graph_lim+1000]); ax_h.set_ylim([0,graph_lim+1000])
ax_h.grid()

ship, = ax_h.plot([],[], 'bo')
time_txt = ax_h.text(0.86,0.7,'', 
                     color= 'C2', fontsize= 40, 
                     alpha= 0.6, transform=ax_h.transAxes)

info_text= '''
Round object fall simulation.\n
The object is slightly asimmetryc.
This generates a certain lift vector
which can be used for steering.\n
Press z or x 
to turn the object 180deg'''
ax_h.text(0.8, 0.45, info_text, transform=ax_h.transAxes)

q_list = []
for i in range (0,5):
    vector, scale, color, label = vectors_list[i]
    q_list.append(ax_h.quiver(s,h, 0, 0,
                              color = color,
                              label= label, alpha = 0.7) )    
quiver_v = q_list[0]
quiver_G = q_list[1]
quiver_L = q_list[2]
quiver_D = q_list[3]
quiver_R = q_list[4]


ax_h.legend(loc = 1)


#%% animating  
  
def gen(): #generates a number of frames if a condition is fullfilled
    global h
    frame = 0
    while h > 50:
        frame += 1
        yield frame

def init ():
    
    ship.set_data([],[])
    time_txt.set_text('')
   
    return ship,time_txt,

def update(frame):
    
    global status, status_DL
    global quiver_v, quiver_G, quiver_L, quiver_D, quiver_R
    
    s,h, vectors_list= gen_data() #
    
    #ax_h.collections.clear()
    
    ship.set_data(s,h)
    
    template = '{:.1f}s'.format(total_time)
    time_txt.set_text(template)
    
    q_list = []
    
    for i in range (0,5):  #Later I see there is a method quiver.set_UVC that could substitute the list thing
        vector, scale, color, label = vectors_list[i]
        q_list.append(ax_h.quiver(s,h,vector[0],vector[1],
                                  scale = scale, color = color,
                                  label= label, alpha = 0.7) )
        
    quiver_v = q_list[0]
    quiver_G = q_list[1]
    quiver_L = q_list[2]
    quiver_D = q_list[3]
    quiver_R = q_list[4]
    
    return ship, time_txt, quiver_v, quiver_G, quiver_L, quiver_D, quiver_R,
    
    
ani = FuncAnimation(fig, update, 
                    init_func=init, 
                    frames = gen, 
                    interval = interval, repeat = False, blit=True)

plt.show()

#%% load arrays for later analysis
statuss = np.load('status.npy')
status_DLs = np.load('status_DL.npy')

