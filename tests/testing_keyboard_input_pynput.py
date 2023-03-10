#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 13:05:32 2021

@author: paco
"""

from pynput import keyboard
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Start variables and "listen" to the keyboard
coef = 1
x = 1


                            


def on_press(key):
    
    global coef
    if key.char == 'a':
        coef = coef + 0.01
    elif key.char == 'z':
        coef = coef - 0.01
    else:
        pass
    print (str(coef),'\n')

listener=keyboard.Listener(on_press=on_press)
listener.start()                   #this start listening the keyboard
#listener.stop() stops listening!!!!   
        
fig= plt.figure()
ax = fig.add_subplot(1,1,1)

def animate (i):
        
    
        
    global x,coef
        
    y = coef * x
    
    ax.set_xlim([0,300]); ax.set_ylim([0,300])
    
    ax.scatter(x,y)
    
    x += 1
            
ani = FuncAnimation(fig, animate, interval = 100, frames = 100)

plt.show()
    
    
       
        
        