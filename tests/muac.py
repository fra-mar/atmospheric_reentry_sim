#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:34:36 2021

@author: paco
"""

def find_angle(v1,v2):
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
    
    return np.rad2deg(angle)

def lift_function (vel_init, vel_after,theta):

    module = lambda v: np.sqrt( np.sum(v**2))
    v_unitary = lambda v: v/module(v)
    
    vel_init_u = v_unitary(vel_init)    #unitary vector
    
    
    vel_after_u =v_unitary(vel_after)
    '''
    v_lift = np.array([ [1e-3,1e-3,3]]) #V_o is the orthogonal lift vector of the capsule
    '''
    v_side = np.cross(vel_init,vel_after)
    v_lift = np.cross(vel_after,v_side)
    v_lift_u = v_unitary (v_lift)
    #Rodriges formula
    v_lift_r_a = v_lift * np.cos(theta) + (np.cross(vel_after_u,v_lift)) * np.sin(theta)
    v_lift_r_b = vel_after_u* (np.dot(vel_after_u[0],v_lift[0]))*(1-np.cos(theta))
    
    v_lift_r = v_lift_r_a + v_lift_r_b
    
    v_lift_r_u = v_unitary(v_lift_r)
    
    incl_lift = find_angle(v_lift_u,v_lift_r_u)
    
    
    
    return v_lift_u,v_lift_r_u,incl_lift