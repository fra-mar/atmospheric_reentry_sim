#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 07:24:28 2021

@author: paco
"""

'''
To transform a distance to geographical degrees.
1 degree at equator = 111319.5 m ( 2*pi*Rearth / 360)
As yo move north/south, that scale kept constant for lat, but for long...
1 degree long = 111319.5 * cos (lat)

Thus given a change in position you calculate the 'horizontal' and 'vert'
components, then apply formulas:

n degrees long = hor/ (cos(lat) * 111319.5)
n degrees lat = vert / 111319.5
'''

import numpy as np

k = 111319.5  #1 degreee equivalence in meters at equator. OBS! NOT RADIANS

long1 = 1 #DEGREES
lat1 = 23 #DEGREES

hor = 1e6  #1000 Km move to the east
ver = 0 #no move N/S

hor = long1 * np.cos( np.deg2rad(lat1) )*k

print ('1 hor degree at {} degrees lat are {:.3f} meters'.format(lat1, hor) )

#%% change in longitude with distance
long1 = 17.64
lat1 = 59.86

hor_distance = 5e5 #500 km to the east

long2 = long1 + hor_distance / (np.cos( np.deg2rad(lat1) ) * k )

print ('Starting at {} lat, {} long, moving {} m to the east gives {:.2f} long'.
      format(lat1, long1, hor_distance,long2) )

#%% change in longitude and latitude with distance in the hor and vert axis
long1 = 17.64
lat1 = 59.86

hor_distance = 5e5 #500 km to the east
vert_distance = 5e4 #50 km to the north

long2 = long1 + hor_distance / (np.cos( np.deg2rad(lat1) ) * k )
lat2 = lat1 + vert_distance / k

print ('Starting at {} lat, {} long, moving {} m to the east and {} m to the north gives {:.2f} lat and  {:.2f} long'.
      format(lat1, long1, hor_distance,vert_distance, lat2, long2) )