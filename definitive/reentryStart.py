#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:37:53 2022

@author: paco

choose an option 
"""
import tkinter as tk
from tkinter import ttk

from subprocess import Popen
#from multiprocessing import Pool, Process
import os

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("800x600+10+10")
        self.title('Atmospheric reentry simulator. v1.3. Start menu')
        self.configure(bg= '#f9cb9c')
        paddings = {'padx': 5, 'pady': 5}
        s= ttk.Style()
        s.configure('TMenubutton', background= '#b0ebf4')
        s.configure('TButton', background= '#e69138', foreground= 'black')
       
        # initialize data
        self.models = ('Profile 1',
                       'Profile 2')
        
        self.ld= tk.DoubleVar(self,  value= 0.4)
        self.weight= tk.DoubleVar(self, value= 2200)
        self.area= tk.DoubleVar(self,  value= 3.8)
        self.cd= tk.DoubleVar(self, value= 1)

        # set up variable
        self.model_var = tk.StringVar(self)
        

        # create widget
        self.create_widgets()
        
        #create entries for mass, area, Cd and L/D 
        
        self.ld= tk.Entry(self, width= 5, textvariable= self.ld)
        self.ld.grid(row= 6, column= 1, sticky=tk.W, **paddings)
        
        self.w= tk.Entry(self, width= 5, textvariable= self.weight)
        self.w.grid(row= 3, column= 1, sticky=tk.W, **paddings)
        
        self.a= tk.Entry(self, width= 5, textvariable= self.area)
        self.a.grid(row= 4, column= 1, sticky=tk.W, **paddings)
        
        self.d= tk.Entry(self, width= 5, textvariable= self.cd)
        self.d.grid(row= 5, column= 1, sticky=tk.W, **paddings)
        

        self.start_button= ttk.Button(self, text='START SIMULATION', 
                                      command= self.start,
                                      state= 'enabled'
                                      )
        self.start_button.grid(row=8, column= 3, 
                               rowspan= 2, 
                               sticky=tk.W, padx=60, pady= 15)
   
    def create_widgets(self):
        # padding for widgets using the grid layout
        paddings = {'padx': 5, 'pady': 5}

        # labels
        label_models = ttk.Label(self,  text='Profile(beta)',
                                 background= '#f9cb9c')
        label_models.grid(column=0, row=0, sticky=tk.W, **paddings)
        
        label_weight= ttk.Label(self, text='Mass (Kg)',
                                 background= '#f9cb9c')
        label_weight.grid(column=0, row=3, sticky=tk.E, **paddings)
        
        label_area= ttk.Label(self, text='Area(m2)',
                                 background= '#f9cb9c')
        label_area.grid(column=0, row=4, sticky=tk.E, **paddings)

        label_d= ttk.Label(self, text='Drag coeff',
                                 background= '#f9cb9c')
        label_d.grid(column=0, row=5, sticky=tk.E, **paddings)
        

        label_ld= ttk.Label(self, text='Lift/Drag ratio',
                                 background= '#f9cb9c')
        label_ld.grid(column=0, row=6, sticky=tk.E, **paddings)
        
        doThis=''' 
        Choose the vehicle parameters on the left. 
        Default values are Soyuz's.

        The simulation starts at aprox 70Km heading
        Kazakhstan, as a Soyuz comming from the ISS

        Use 'z' and 'x' to change the bank angle
        (0 deg at start). Positive values means
        port banking. Negative: starboard banking
        
        Educated guesses for different vehicles 
        at reentry are:

        \t\tM\tA\tCd\tL/D 
        Soyuz\t2.2\t3.8\t1.1\t0.4
        Apollo\t5.5\t12\t1.26\t0.36
        Space Shuttle\t100\t245\t0.9\t1.1

        M: mass(t)  A: area(m2)
        Cd: Drag coefficient
        L/D: Lift to Drag ratio
        '''
        label_doThis= ttk.Label(self, text= doThis,
                               background= '#f9cb6f',
                               foreground='black',
                               font= ('Cambria',10),
                               justify= 'left')
        label_doThis.grid(column= 3, row= 3,
                          sticky= 'e',
                          columnspan= 2,
                          rowspan= 4, padx= 50)
      
        info='Free to distribute. Francisco Martinez Torrente. May 2023'
       
        label_info= tk.Label(self,  text= info,
                                 background= '#f9cb9c',
                                 foreground= '#38761d',
                                 font= ('Cambria',8))
        label_info.grid(column=0, 
                        columnspan=5,rowspan=2, row=10, sticky=tk.SW, **paddings)

        # option menus
        
        model_menu = ttk.OptionMenu(
            self,
            self.model_var,
            self.models[0],
            *self.models)
        model_menu.grid(column=1, row=0, sticky=tk.W, **paddings)
        
        # output label
        self.output_label = ttk.Label(self, foreground='red')
        self.output_label.grid(column=1, row=7, 
                               columnspan= 2, 
                               sticky=tk.W, **paddings)

    
    def start(self):  
        
        self.ld= self.ld.get()
        self.w= self.weight.get()
        self.a= self.area.get()
        self.d= self.cd.get()
        pwd=os.getcwd()
        if os.path.isdir(os.path.join(pwd, 'lib_files'))==False:
            os.mkdir('lib_files')
        else:
            pass

        file_path= os.path.join(pwd,'lib_files','init_params.txt')
        print (file_path)
        with open(file_path, 'w') as f:
            f.write(f'{self.w},{self.a},{self.d},{self.ld}')
        
        #to be used for final compilation linux
        #path= os.path.join('dist','CCIP_main','CCIP_main')  
        #Popen([f'{path}'])
        
        #to be used for final compilation windows
        #path= os.path.join('dist','CCIP_main','CCIP_main.exe') 
 	#Popen([f'{path}']

        #to be used while developing
        path= ['python','reentry_v2.py']
        Popen(path)
        
       
            

if __name__ == "__main__":
    app = App()
    
    app.mainloop()
