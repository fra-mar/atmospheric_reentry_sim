#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 05:44:01 2021

@author: paco
"""


import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere

# Data for spacific altitude
sealevel = Atmosphere(0)
print ('Density at 0 m: {} Kg/m3'.format (sealevel.density))

#%% Make an atmosphere object
heights = np.linspace(-5e3, 80e3, num=1000)
atmosphere = Atmosphere(heights)

# Make plot
plt.plot(atmosphere.temperature_in_celsius, heights/1000)
plt.ylabel('Height [km]')
plt.xlabel('Temperature [°C]')
plt.grid()
plt.show()
