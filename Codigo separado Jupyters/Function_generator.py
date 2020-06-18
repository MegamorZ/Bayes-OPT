# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:19:24 2020

Random Function generator

@author: megam
"""
import numpy as np

np.random.seed(46)

class RandomFunc:
    def __init__(self,dim_weight):
        self.A = np.random.randint(-10,10)
        self.b = np.random.randint(-10,10)
        #self.c = np.random.randint(-10,10)
    
    def Quad2D(self,x):
        return self.a*x[0]**2+self.b*x[1]**2+self.c*