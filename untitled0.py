# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 08:05:54 2021

@author: Emmanuel_Ledezma_H
"""

import tkinter as tk
from Trainig import training

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")
        
        self.train = tk.Button(self)
        self.train["text"] = "Train Model"
        self.train["command"] = self.call_training_model
        self.train.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")
        
    def call_training_model(self):
        training
        

root = tk.Tk()
app = Application(master=root)

#
# here are method calls to the window manager class
#
app.master.title("Air Canvas applicat")
app.master.geometry('600x300')

#Start the program
app.mainloop()