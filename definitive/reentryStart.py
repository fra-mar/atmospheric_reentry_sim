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
        self.geometry("800x300+10+10")
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
        
        # set up variable
        self.model_var = tk.StringVar(self)
        

        # create widget
        self.create_widgets()
        
        #create an entry for age, t.ex.
        
        self.ld= tk.Entry(self, width= 5, textvariable= self.ld)
        self.ld.grid(row= 4, column= 1, sticky=tk.W, **paddings)
        
        self.w= tk.Entry(self, width= 5, textvariable= self.weight)
        self.w.grid(row= 3, column= 1, sticky=tk.W, **paddings)
        
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
        
        label_ld= ttk.Label(self, text='Lift/Drag ratio',
                                 background= '#f9cb9c')
        label_ld.grid(column=0, row=4, sticky=tk.E, **paddings)
        
        doThis=''' 
        As a reference, approximate values for 
        different vehicles at reentry are:

        \t\tLift/Drag ratio
        Soyuz\t0.4
        Apollo\t0.3
        Space Shuttle\t1.0 
        Airliner\t17.0
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
       
        pwd=os.getcwd()
        if os.path.isdir(os.path.join(pwd, 'lib_files'))==False:
            os.mkdir('lib_files')
        else:
            pass

        file_path= os.path.join(pwd,'lib_files','init_params.txt')
        print (file_path)
        with open(file_path, 'w') as f:
            f.write(f'{self.w},{self.ld}')
        
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
