#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:54:32 2021

@author: paco
"""

'''This script has auxilliary functions for reentry script

'''
import numpy as np

#%% lambdas and functions

module = lambda v: np.sqrt( np.sum(v**2))
v_unitary = lambda v: v/module(v)


#%% Calculates lift_vector, first a reference that always points upwards, normal to velocity
# then rotated over the velocity vector according to Rodriges formula

def lift_vector (vel_after_u, theta):
    #if lift vector is perpendicular to vel vector, then dot product must be 0.
    #if x and y components should be equal for vel and lift vectors, then is the z component that should be calculated
    v_lift_x = vel_after_u[0][0]
    v_lift_y = vel_after_u[0][1]
    v_lift_z = abs(  ( -vel_after_u[0][0]**2 - vel_after_u[0][1]**2) / vel_after_u[0][2]   )  #abs to ensure always positve
    
    v_lift_u = v_unitary( np.array( [ [v_lift_x,v_lift_y,v_lift_z]]))
    
    #Rodriges formula to add a rotation around the velocity vector. 
    v_lift_r_a = v_lift_u * np.cos(theta) + (np.cross(vel_after_u,v_lift_u)) * np.sin(theta)
    v_lift_r_b = vel_after_u * (np.dot(vel_after_u[0],v_lift_u[0]))*(1-np.cos(theta))
    v_lift_r = v_lift_r_a + v_lift_r_b
    
    v_lift_r_u = v_unitary(v_lift_r)
    
    return v_lift_r_u

#%% air density as a function of altitude
#based on https://nptel.ac.in/content/storage2/courses/101106041/Chapter%202%20-Lecture%205%2020-12-2011.pdf
def air_density(h):
    if h <= 11e3:      #11e3 m limit for troposphere
        rho = 1.225* ( ( 1 - 2.2588e-5 * h) ** 4.25588)
    elif h > 11e3:   #start of stratosphere, a different equation applies
        rho = 0.36391* np.exp( -0.000157688 *( h-11e3))
    elif h > 20e3:
        rho = 0.08803* ( (1+0.000004616*(h-20000))**(-35.1632) )
    return rho

#%% gravity force calculator
def gravity_calc (h,m):
    
    grav_constant = 6.67408*1e-11 #units m3 kg-1 s-2
    radius_earth = 6.371*1e6   #units in m
    earth_mass = 5.9722*1e24 #units in kg
    Gy = grav_constant * earth_mass * m / ( (radius_earth + h)**2 )
    G = np.array( [[0,0,-Gy]])
    return G

#%% area for the Drag force calculation.
'''simulates parachutes deploying under 10.7km altitude
real soyuz deploys brake parachute at 10.7Km and main at 8.5Km. 

'''
    
def area_drag(h):
    if h>= 10.7e3 :
        A_drag = 3.8
        par_status = 'PACKED'
    elif h<10.7e3 and h >=  9e3:
        A_drag = 25* (9e3/h)**8 #parenthesis and power for 'slow' release
        par_status = 'BRAKE'
    elif h <9e3:
        A_drag = 100 * (8.2e3/h)**8
        par_status = 'MAIN'
    return A_drag, par_status
#%% Dictionary with landing sites
landing_sites = {'0':(69.30,47.10),
                 '1':(49.10,53.00),
                 '2':(76.30,53.00),
                 '3':(75.50,48.10),
                 '4':(60.50,42.30),
                 '5':(58.50,47.10),
                 '6':(48.00,49.70),
                 '7':(48.00,52.00)}



