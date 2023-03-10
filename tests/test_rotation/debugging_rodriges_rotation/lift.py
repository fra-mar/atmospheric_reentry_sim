#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:54:32 2021

@author: paco
"""

'''So at last I found out that a lift vector can be calculated as cross product of 
vel_init x after (cross) and then cross product again of cross x vel_after.
 To rotate the lift vecto around the velocity vector 
 (let's say vel_after a brilliant formula can be
used -Rodriges formula-)

'''


import numpy as np



#%% lambdas and functions

module = lambda v: np.sqrt( np.sum(v**2))
v_unitary = lambda v: v/module(v)
def find_angle(v1,v2):
    
    try:
        module = lambda v: np.sqrt( np.sum(v**2))
        modules_product = module(v1)*module(v2)
        dot_product =  np.dot(v1[0],v2[0])
        
        if modules_product > 1e-15:#  and dot_product >1e-15:
            angle = np.arccos( dot_product / modules_product  )
        else:
            angle = 0.0
            print ('oooops')
        angle = np.arccos( dot_product / modules_product  )
        
        cross = np.cross ( v1, v2)
        dot = np.dot (cross[0], np.array( [0.,1.,0.]))
        if dot <0:
            angle = -angle
    except:
        angle = 0.0
        
    return np.rad2deg(angle)

#%% start vectors and angles
'''
x_u= np.array( [[1,0.0,0.0]])

y_u = np.array( [[0.,1.,0.]])

z_u = np.array( [[0.,0.,1]])

alpha, beta, gamma = 0,0,0
'''
#%%
def lift_vector (vel_after_u, theta):
    #if lift vector is perpendicular to vel vector, then dot product must be 0.
    #if x and y components should be equal for vel and lift vectors, then is the z component that should be calculated
    v_lift_x = vel_after_u[0][0]
    v_lift_y = vel_after_u[0][1]
    v_lift_z =( -vel_after_u[0][0]**2 - vel_after_u[0][1]**2) / vel_after_u[0][2]   
    
    v_lift_u = v_unitary( np.array( [ [v_lift_x,v_lift_y,v_lift_z]]))
    
    #Rodriges formula to add a rotation around the velocity vector. 
    v_lift_r_a = v_lift_u * np.cos(theta) + (np.cross(vel_after_u,v_lift_u)) * np.sin(theta)
    v_lift_r_b = vel_after_u * (np.dot(vel_after_u[0],v_lift_u[0]))*(1-np.cos(theta))
    v_lift_r = v_lift_r_a + v_lift_r_b
    
    v_lift_r_u = v_unitary(v_lift_r)
    
    return v_lift_u, v_lift_r_u


#%% initial and final vector and their components

def lift_function (vel_init, vel_after,theta):

    module = lambda v: np.sqrt( np.sum(v**2))
    v_unitary = lambda v: v/module(v)
    
    vel_init_u = v_unitary(vel_init)    #unitary vector
    
    
    vel_after_u =v_unitary(vel_after)
   
    v_side = np.cross(vel_init,vel_after)
    v_lift = np.cross(vel_after,v_side)
    v_lift_u = v_unitary (v_lift)
    #Rodriges formula
    v_lift_r_a = v_lift * np.cos(theta) + (np.cross(vel_after_u,v_lift)) * np.sin(theta)
    v_lift_r_b = vel_after_u* (np.dot(vel_after_u[0],v_lift[0]))*(1-np.cos(theta))
    
    v_lift_r = v_lift_r_a + v_lift_r_b
    
    v_lift_r_u = v_unitary(v_lift_r)
    
    #incl_lift = find_angle(v_lift_u,v_lift_r_u)
    incl_lift = 0.0
    
    
    
    return v_lift_u,v_lift_r_u,incl_lift


#%% air density as a function of altitude
#based on https://nptel.ac.in/content/storage2/courses/101106041/Chapter%202%20-Lecture%205%2020-12-2011.pdf
def air_density(h):
    if h <= 11e3:      #11e3 m limit for troposphere
        rho = 1.225* ( ( 1 - 2.2588e-5 * h) ** 4.25588)
    elif h > 11e3:   #start of stratosphere, a different equation applies
        rho = 0.36391* np.exp( -0.000157688 *( h-11e3))
    return rho
