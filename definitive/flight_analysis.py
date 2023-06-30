#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:01:17 2021

@author: paco
"""
import numpy as np
import os
from matplotlib import pyplot as plt

from lift import landing_sites

#%% Plotting flight analysis

path= os.path.join(os.getcwd(),'lib_files','flight_data.csv')
flight_data = np.genfromtxt (path, delimiter = ',', skip_header=1)

fig2 = plt.figure('Flight analysis',figsize = (16,8), facecolor = '#eeeeee')
fig2.subplots_adjust(wspace=0.3, hspace= 0.3)

ax2 = fig2.add_gridspec(2,2)

ax2_h = fig2.add_subplot(ax2[0,0])
ax2_h.scatter(flight_data[:,0], flight_data[:,3]/1000, marker = '^', 
              s = 4, c = 'green')

ax2_h.set_xlabel('Time(s)',fontsize = 12)
ax2_h.set_ylabel('Altitude (Km)',fontsize = 12)
ax2_h.grid(axis  = 'y')


ax2_yx = fig2.add_subplot(ax2[0,1])
limit_xmax = flight_data[:,1].max()*1.02
limit_xmin = flight_data[:,1].min()*1.02
limit_ymax = flight_data[:,2].max()*1.02
limit_ymin = flight_data[:,2].min()*1.02

ax2_yx.set_xlim([limit_xmin,limit_xmax])
ax2_yx.set_ylim([limit_ymin,limit_ymax])
ax2_yx.set_title('Path in horizontal plane', fontsize = 12)
ax2_yx.set_ylabel('Latitude (deg)',fontsize = 12)
ax2_yx.set_xlabel('Longitude (deg)',fontsize = 12)
ax2_yx.grid (axis='both')

ax2_yx.scatter(flight_data[:,1], flight_data[:,2], s = 6)

ax2_g_s = fig2.add_subplot( ax2[1,0])
ax2_g_s.scatter(flight_data[:,0], flight_data[:,4], s = 6, c = 'black' )
ax2_g_s.set_xlabel('Time(s)',fontsize = 12)
ax2_g_s.set_ylabel('G forces',fontsize = 12)
ax2_g_s.grid(axis  = 'y')

#distance to nearest nearest planned landing site
distances = []
long = flight_data[-1,1]
lat = flight_data[-1,2]
k = 111316.5 #meters/deg at the equator
total_time = flight_data[-1,0]
for p in landing_sites:
    lng, ltd = landing_sites.get(p)
    distance_deg = np.sqrt( (long-lng)**2 + (lat-ltd)**2)  
    distance_m = distance_deg * k
    distances.append(distance_m)
distances = np.array(distances)
closer_distance ='{:.1f}'.format( distances.min()/1000)

#flight duration
duration = '{:.1f}'.format (total_time/60)

ax2_data = fig2.add_subplot(ax2[1,1])
ax2_data.xaxis.set_visible(False); ax2_data.yaxis.set_visible(False)
ax2_data.text( 0.2, 0.4, 'Time from 70Km to 8 Km (min):', fontsize =12)
ax2_data.text( 0.2, 0.6, 'Nearest landing site (Km):', fontsize = 12)
ax2_data.text(0.7,0.4, duration, fontsize = 20)
ax2_data.text(0.7,0.6, closer_distance, fontsize = 20)

plt.title('Flight analysis')
plt.show()
#%% Save flight data

#flight data: time, long, lat, altitude, g force, rho, Cd, vel, lift, drag, gravity, resultant

np.savetxt('flight_data.csv', flight_data, delimiter=',', 
           header = 'Time,Long,Lat,Alt,g_s,rho,vel,Cd,Lift,Drag,Gravity,Resultant')

