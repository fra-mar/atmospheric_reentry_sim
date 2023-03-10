#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:54:32 2021

@author: paco
"""

'''testing linear transformatons, according to 3eyeBlue video series linear algebra
https://www.youtube.com/watch?v=rHLEWRxRGiM
'''
import numpy as np


#%% lambdas and functions
to_rad = lambda d: np.pi*d / 180
to_deg = lambda r: r*180/np.pi


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
        
        cross = np.cross ( v1, v2)
        dot = np.dot (cross[0], np.array( [0.,1.,0.]))
        if dot <0:
            angle = -angle
    except:
        angle = 0.0
        
    return angle

def xyz_components(v):
    v_x = np.array( [ [v[0][0], 1e-4,1e-4]])
    v_y = np.array( [ [ 1e-4, v[0][1], 1e-4]] )
    v_z = np.array( [ [ 1e-4, 1e-4, v[0][2]] ])
    return v_x, v_y, v_z


def t_matrix_ax(x_u,y_u,z_u):
    t_matrix_ax = np.array([[x_u[0][0], y_u[0][0], z_u[0][0]],
                         [x_u[0][1], y_u[0][1], z_u[0][1]],
                         [x_u[0][2], y_u[0][2], z_u[0][2]] ])
    return t_matrix_ax


#%% calculation of alpha,beta,gamma after change in velocity vector


    
def lift_function_beta (vel_original, vel_after, v_lift_original, theta):   
     
    vel_after_u =v_unitary(vel_after)
    
    vel_after_u_x, vel_after_u_y, vel_after_u_z = xyz_components(vel_after_u)

    vel_original_u = v_unitary(vel_original)    #unitary vector
    
    vel_original_u_x, vel_original_u_y, vel_original_u_z = xyz_components(vel_original_u)
    
    v_lift_original_u = v_unitary( v_lift_original) 
    
    print ('Angle vel_original,vel_after : {:.2f} degrees'.format(to_deg(find_angle(vel_original,vel_after))))
    
    
    # rotation around the z axis, i.e. in plane xy, i.e. alpha angle. If block to decide if adds or substracts
    proj_xy_original_u = vel_original_u_x + vel_original_u_y
    proj_xy_after_u = vel_after_u_x + vel_after_u_y
    alpha = find_angle(proj_xy_after_u,proj_xy_original_u)
    
    
    # rotation around the x axis, i.e. in plane yz, i.e. beta angle
    proj_yz_original_u = vel_original_u_y + vel_original_u_z
    proj_yz_after_u = vel_after_u_y + vel_after_u_z
    beta = find_angle(proj_yz_after_u,proj_yz_original_u)
   
    
        
    # rotation around the y axis, i.e. in plane xz, i.e. gamma angle   
    proj_xz_original_u = vel_original_u_x + vel_original_u_z
    proj_xz_after_u = vel_after_u_x + vel_after_u_z
    gamma = find_angle(proj_xz_after_u,proj_xz_original_u)
  
    
    
    print ('degrees alpha {:.3f}  beta{:.3f}  gamma{:.3f}'.
           format(to_deg(alpha), to_deg(beta), to_deg(gamma) ))
    
    #%% successive rotations
     
    #round axis z
    
    z_u_z = np.array( [[0.,0.,1]])
    x_u_z = np.array (  [ [np.cos(alpha), np.sin(alpha), 0] ]) 
    y_u_z = np.array (  [ [-np.sin(alpha), np.cos(alpha), 0] ])
    
    t_matrix_z = t_matrix_ax( x_u_z, y_u_z, z_u_z)
    
    #round axis x
    
    x_u_x = np.array( [[1,0.0,0.0]])
    y_u_x = np.array ( [ [0, np.cos(beta), np.sin(beta)] ])
    z_u_x = np.array ( [ [0, -np.sin(beta), np.cos(beta)] ])
    
    t_matrix_x = t_matrix_ax( x_u_x, y_u_x, z_u_x)
    
    #round axis y
    y_u_y = np.array( [[0.,1.,0.]])
    x_u_y = np.array ( [ [np.cos(gamma), 0, -np.sin(gamma)] ])
    z_u_y = np.array ( [ [np.sin(gamma),0,  np.cos(gamma)] ])
    
    t_matrix_y = t_matrix_ax( x_u_y, y_u_y, z_u_y)
    
    
    #%% all three rotation axis combined
    
    t_matrix_rot = np.matmul( np.matmul(t_matrix_x,t_matrix_z) , t_matrix_y)
    
    '''
    t_x = np.array( [t_matrix_rot[:,0]])
    t_y = np.array( [t_matrix_rot[:,1]])
    t_z = np.array( [t_matrix_rot[:,2]])
    '''
    #%% calculating effect of velocity vector change in lift vector
    
    v_lift_transformed = np.matmul(t_matrix_rot, v_lift_original.T).reshape((1,3))
    
    v_lift_t_u = v_unitary(v_lift_transformed)
    
    incl_lift = 0
    
    return v_lift_original_u, v_lift_t_u, incl_lift
    
    
 





