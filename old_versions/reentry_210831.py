#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Please read README.txt in...

Reentry simulates the path of a Soyuz capsule through the atmosphere on its 
way back to Earth. 
The goals were to do it simple, and learn myself more about physics and programming.
Thatś why only numpu, matplotlib and pynput libraries are needed.
The Soyuz has an offset center of mass that makes it tilt. This assymmetric 
configuration generates som lift that can be used to change her direction 
by means of rotation. Letters z (counterclockwise, positve degrees) and x (clockwise,
negative degrees) are used to rotate the capsule.

Some parameters are stored in the (numpy) array flight_data that can be saved at the 
end of a simulation for further analysis.

The simulation can be interrupted by closing the figure and then clicking ctr-C

If you are running from IDLE itś possible you can visualize and save flight data
after interruption by running the last 2 cells.

Regarding initial drag coefficient and lift to drag ration I recommend default
for 'realism' or Cd=0.5 L/D=1.5 for longer, more controllable flights.

Cd changes with a variety of factors. For simplicity, I assume a constant angle 
of attack, and a increased Cd with lower air density, as a very simplified 
surrogate of Reynolds number.

Regarding vectors axis, view can be changed by dragging with the mouse

'''
import numpy as np

from matplotlib import pyplot as plt

from mpl_toolkits import mplot3d

from pynput import keyboard

from lift import  lift_vector, air_density, gravity_calc, area_drag, landing_sites


#%% Ask user what Cd and L/D ratio

Cd = input ('\nWhat drag coefficient? (rec 0.3-2, 1.26 deffault): ') 

LD_ratio = input ('\nWhat lift to drag ratio? (rec 0.3-2, 0.4 deffault): ') 

if Cd=='':
    Cd = 1.26
    
if LD_ratio == '':
    LD_ratio = 0.4

Cd, LD_ratio = float (Cd), float(LD_ratio)

#Draws equation to correct Cd after rho.
rho_min, rho_max = air_density(1e4), air_density(7e4)

Cd_high = Cd *1.4 #at higher atmosphere Cd will be 40% higher compared to lower atm. 

# asuming equation in the form y = Ax^n, i.e., Cd = k * rho^k ...
n = (np.log10(Cd) - np.log10(Cd_high)) / (np.log10(rho_min) - np.log10(rho_max) )

log_A = n * (-np.log10(rho_max)) + np.log10(Cd_high)

#%% Axes definiton

plt.ion()

fig = plt.figure('Soyuz, reentry path', figsize = (16,5))

ax = fig.add_gridspec(3,5)

plt.subplots_adjust(wspace = 0.8)

ax_xy = fig.add_subplot(ax[0:,2:]) #axis for map view

img = 'Kazakhstan_map.png'

map = plt.imread(img)

extents = (46.000,88.000,40.000,56.000)

ax_xy.imshow(map, aspect = 'auto') 

#axis that will show vectors

ax_vectors =  fig.add_subplot(ax[:,0:2],projection = '3d')

ax_vectors.view_init(10,30)

x_ax_vectors = np.linspace(-7,7,7) #this helps defines axis size
y_ax_vectors = np.linspace(-7,7,7)
z_ax_vectors = y_ax_vectors.copy()

X,Y = np.meshgrid(x_ax_vectors,y_ax_vectors)

Z=0*X + 0*Y

YY,ZZ = np.meshgrid ( y_ax_vectors,z_ax_vectors)

XX = 0*Y + 0*Z

ax_vectors.invert_xaxis() #this inverts x axis, otherwise graphically wrong.

#%% Lambdas and coversions

module = lambda v: np.sqrt( np.sum(v**2)) 

antivector_u = lambda v: -1 * v/module(v)

v_unitary = lambda v: v/module(v)

k = 111316.5 #meters/deg at the equator

#%% initial variables

A = 3.8 ; m = 2300.0; rot = 1 #projected area, mass = 2.3 tons

h = 70.1e3 #altitud in meters

rho = air_density(h) #air density in Kg/m3

vel_init = np.array([[4720.7231,5956.0702,-1.01]]) # for orbit incl 51.6deg

vel_after = np.array([[4720.7231,5956.0702,-2.01]]) 

theta = 0 #angle in degrees. angle intended to turn the lift vector about velocity axis

lat_init = 40.0; long_init = np.random.randint(46,75)

lat, long = lat_init, long_init

total_time = 0

#flight data: time, long, lat, altitude, g force, rho, vel, Cd, lift, drag, gravity, resultant
flight_data = np.array ([0., long,lat, h, 0,0,module(vel_after),Cd,0,0,0,0]) 

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

#%% Main loop

while h > 8e3:  #under 8000m nothing happens, that's why sim stops here
    
    ax_xy.clear()
    ax_vectors.clear() #clear the axis
       
    if rot == 1 or rot == -1: #change Soyuz's rotation angle
        
        if rot == 1:
            theta += np.deg2rad(4)
            if theta>np.pi: #change sig if over 180 deg
                theta = -np.pi - (theta-np.pi)
        elif rot == -1:
            theta -=np.deg2rad(4)
            if theta<-np.pi:
                theta = np.pi - (theta-(-np.pi))
        
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
    
    total_time += t
    
    #Calculates heading
    heading = np.rad2deg( np.arctan2 (y_displ,x_displ) )
    if heading<0:
        heading = 360 + heading
    
#%% Plotting map and vectors
        
    ax_xy.set_title ('Primary landing site (star). Secondaries (triangles)')
    y_ticks = [y+40 for y in range (3,16,3)]
    x_ticks = [x+45 for x in range (3,42,3)]
    ax_xy.set_xticks(x_ticks); ax_xy.set_yticks(y_ticks)
    ax_xy.imshow(map,extent = extents,aspect  = 'auto', alpha = 0.5)
    ax_xy.grid()
    
    #pins primary and secondary landing sites
    ax_xy.scatter(69.30, 47.10, marker = '*', color = 'red') 
    for l in range(1, len(landing_sites)):
        c = landing_sites.get(str(l))
        ax_xy.scatter (c[0], c[1], marker = 'v', color = 'gray')
    
    #pins the Soyuz in the map 
    ax_xy.scatter(long,lat,color = 'b')    
    
    #Data display in the map
    ground_speed = '{:.2f} m/s'.format ( module ( np.array([vel_after[0][0],vel_after[0][1]])))
    vertical_speed = '{:.2f} m/s'.format (vel_after[0][2])
    g_string = '{:.2f}'.format(g_s)
    cap_rot_str = '{:3.2f} deg'. format(np.rad2deg(theta))
    lat_str = 'Lat: {:.3f}'.format(lat)
    long_str = 'Long: {:.3f}'. format(long)
    heading_str = 'Heading: {:.2f} deg'.format(heading)
    
    ax_xy.text(80,55, 'Altitude') 
    ax_xy.text(80,54, '{:.2f} m'.format(h), fontsize = 'large')
    
    ax_xy.text(80,52.5, 'GroundSpeed')
    ax_xy.text(80,51.5, ground_speed, fontsize = 'large')
    
    ax_xy.text(80,50, 'VerticalSpeed')
    ax_xy.text(80,49, vertical_speed, fontsize = 'large')
    
    ax_xy.text(80,47.5, 'G forces')
    ax_xy.text(80,46.5, g_string, fontsize = 'large')
    
    ax_xy.text(80,45, 'CapsuleRotation')
    ax_xy.text(80,44, cap_rot_str, fontsize = 'large')
    
    ax_xy.text(80,42.5, 'Parachute')
    ax_xy.text(80,41.5, par_status, fontsize = 'large')
    
    ax_xy.text(48,55, lat_str)
    ax_xy.text(57,55, long_str)
    ax_xy.text(66,55, heading_str)
    
    #vectors plot.
    ax_vectors.set_title ('Drag axis with mouse to change view', fontsize = 'small')
    ax_vectors.text(0, -9, -8, 'W',)
    ax_vectors.text(0, 9, -8, 'E',)
    ax_vectors.text(-9, 0, -8, 'S',)
    ax_vectors.text(9, 0, -8, 'N',)
    
    #planes for better vector visualization
    surf1 = ax_vectors.plot_surface(X, Y, Z, alpha = 0.1) 
    surf2 = ax_vectors.plot_surface (XX,YY,ZZ, alpha = 0.1)
    ax_vectors.set_xticklabels([])
    ax_vectors.set_yticklabels([])
    ax_vectors.set_zticklabels([])
    ax_vectors.invert_xaxis()
    
    vectors = [ (vel_after*5,'red','Velocity (m/s)'),
               (G, 'black', 'Gravity'),(R, 'green', 'Resultant'), 
               (D, 'orange', 'Drag'),(L, 'blue', 'Lift')] 
    
    for vv,cc, labels in vectors:
        ax_vectors.quiver(0,0,0,vv[0][0],vv[0][1],vv[0][2], 
                          length = 0.00015, color = cc, label = labels)
        
    ax_vectors.legend(loc = 2,ncol = 3,  mode = 'expand', fontsize = 'small')
    
    plt.pause(0.01) #to prevent software crash 
    
    #prepare variables for next round in the loop and store data in array
    vel_init = vel_after
    rot = 0
    flight_data = np.vstack ((flight_data,
                              [total_time,long,lat, h, g_s,rho,v_mod,Cd,
                              L_mod,D_mod,
                              module(G),module(R)]))
    
    

#%% Plotting flight analysis

fig2 = plt.figure('Flight analysis',figsize = (8,8))
fig2.subplots_adjust(wspace=0.3)

ax2 = fig2.add_gridspec(2,2)

ax2_h = fig2.add_subplot(ax2[0,0])
ax2_h.scatter(flight_data[:,0], flight_data[:,3]/1000, marker = '^', 
              s = 4, c = 'green')
ax2_h.set_xticklabels([])
ax2_h.set_xlabel('Iterations',fontsize = 'small')
ax2_h.set_ylabel('Altitude (Km)',fontsize = 'small')
ax2_h.grid(axis  = 'y')


ax2_yx = fig2.add_subplot(ax2[0,1])
limit_xmax = flight_data[:,1].max()*1.02
limit_xmin = flight_data[:,1].min()*1.02
limit_ymax = flight_data[:,2].max()*1.02
limit_ymin = flight_data[:,2].min()*1.02

ax2_yx.set_xlim([limit_xmin,limit_xmax])
ax2_yx.set_ylim([limit_ymin,limit_ymax])
ax2_yx.set_title('Path in horizontal plane', fontsize = 'medium')
ax2_yx.set_ylabel('Latitude (deg)',fontsize = 'small')
ax2_yx.set_xlabel('Longitude (deg)',fontsize = 'small')
ax2_yx.grid (axis='both')

ax2_yx.scatter(flight_data[:,1], flight_data[:,2], s = 6)

ax2_g_s = fig2.add_subplot( ax2[1,0])
ax2_g_s.scatter(flight_data[:,0], flight_data[:,4], s = 6, c = 'black' )
ax2_g_s.set_xticklabels([])
ax2_g_s.set_xlabel('Iterations',fontsize = 'small')
ax2_g_s.set_ylabel('G forces',fontsize = 'small')
ax2_g_s.grid(axis  = 'y')

#distance to nearest nearest planned landing site
distances = []
for p in landing_sites:
    lng, ltd = landing_sites.get(p)
    distance_deg = np.sqrt( (long-lng)**2 + (lat-ltd)**2)  
    distance_m = distance_deg * k
    distances.append(distance_m)
distances = np.array(distances)
closer_distance ='Nearest landing site: {:.1f} Km.'.format( distances.min()/1000)

#flight duration
duration = 'Time from 70Km to 8 Km {:.1f} min.'.format (total_time/60)

ax2_data = fig2.add_subplot(ax2[1,1])
ax2_data.xaxis.set_visible(False); ax2_data.yaxis.set_visible(False)
ax2_data.text(0.05,0.4, duration)
ax2_data.text(0.05,0.6, closer_distance)

plt.show()
#%% Save flight data

#flight data: time, long, lat, altitude, g force, rho, Cd, vel, lift, drag, gravity, resultant

np.savetxt('flight_data.csv', flight_data, delimiter=',', 
           header = 'Time,Long,Lat,Alt,g_s,rho,vel,Cd,Lift,Drag,Gravity,Resultant')
