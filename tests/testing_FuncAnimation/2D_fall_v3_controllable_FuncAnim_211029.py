#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:49:37 2021

@author: paco



Debugging 2D_fall_animated
"""
import numpy as np

from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation

from pynput import keyboard


#%% very initial parameters
'''
print ('\n2.2 tons space capsule, please set altitud and horizontal speed\n')

h = float( input ('Altitud in meters: ')  )
v_x = float( input ('Horizontal speed: ')  )
'''
h = 3e3
v_x = 300

interval = 50 #interval between frames in MILISECONDS
t =  0.1 # interval between calculations (of v, h, s...)
time_acceleration = t/(interval*1e-3)
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

v = np.array([v_x,0.1]) # vx 2 m/s backwards and 4 m/s upwards

v_mod = module (v)

G = np.array([0,-9.8])*m # G-FORCE

D_mod = 0.5 * (v_mod**2)*A*Cd*rho

D = antivector_u (v) * D_mod

L_mod = 0.5 * (v_mod**2)*A_lift*Cl*rho

L = ortovector_u (v) * L_mod * rot

R = G + D + L

a = R/m

s = 0

graph_lim = h #meters

total_time = 0


status = np.array ([total_time, v[0],v[1],0,0, h])

status_DL = np.array( [ total_time,D[0],D[1],L[0],L[1]] )

#%% Main loop

def gen_data():

    global t, s, h, v, v_mod,  total_time, status, status_DL
    
    D_mod = 0.5 * (v_mod**2)*A*Cd*rho   #calculates drag
    
    D = antivector_u(v) * D_mod

    L_mod = 0.5 * (v_mod**2)*A_lift*Cl*rho    #calculates lift

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
    
    np.save('status.npy', status)
    np.save('status_DL.npy', status_DL)
    
    total_time += t
    print (status[-1,:2])
    
    return s, h

#%% Axes definiton

fig = plt.figure('fall analysis', figsize = (6,6))



ax_h = fig.add_subplot(1,1,1)

ax_h.set_title ('Time acceleration: {:.1f}'.format(time_acceleration))
ax_h.set_xlim([-50,graph_lim+1000]); ax_h.set_ylim([0,graph_lim+1000])
ax_h.grid()

ship, = ax_h.plot([],[], 'bo')
time_txt = ax_h.text(0.8,0.9,'', fontsize= 12, transform=ax_h.transAxes)

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
    
    s,h= gen_data() #
    
    ship.set_data(s,h)
    
    template = 'Time:{:.1f}s'.format(total_time)
    time_txt.set_text(template)
    
    return ship, time_txt,
    
ani = FuncAnimation(fig, update, frames = gen, interval = interval,
                    init_func=init, repeat = False, blit=True)
plt.show()

#%% load arrays for later analysis
statuss = np.load('status.npy')
status_DLs = np.load('status_DL.npy')



    
    