# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:14:04 2020

@author: megam
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import Acquisition_Functions


np.random.seed(46)

def Global_Minima(Fun,bounds,method="L-BFGS-B", nseeds=10,*args):
        Fmin=None
        #fun= lambda x: -fun(x)
        bds=np.array(bounds)
        ndim = bds.shape[1]
        x_seeds = np.random.uniform(0, 1,size=(nseeds, ndim))
        for dim in [0,1,ndim-1]:
            x_seeds[:,dim]= x_seeds[:,dim]*(bds[dim,1]-bds[dim,0])+bds[dim,0]
        
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res =minimize(Fun,x0=x_try,args=args,method="L-BFGS-B",bounds=bds)
            
    #         Store it if better than previous minimum(maximum).
            if Fmin is None or  res.fun < Fmin:
                x_Min = res.x
                Fmin = res.fun 
                
        return x_Min, Fmin

def Global_Maxima(Fun,bounds,method="L-BFGS-B", nseeds=10,*args):
        def mfun(x):
            return -Fun(x)
        Fmax=None
        #fun= lambda x: -fun(x)
        bds=np.array(bounds)
        ndim = bds.shape[1]
        x_seeds = np.random.uniform(0, 1,size=(nseeds, ndim))
        for dim in [0,1,ndim-1]:
            x_seeds[:,dim]= x_seeds[:,dim]*(bds[dim,1]-bds[dim,0])+bds[dim,0]
        
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res =minimize(mfun,x0=x_try,args=args,method="L-BFGS-B",bounds=bds)
            
    #         Store it if better than previous minimum(maximum).
            if Fmax is None or  -res.fun >= Fmax:
                x_Max = res.x
                Fmax = -res.fun 
                
        return x_Max, Fmax


class Bay_Opt:
    def __init__(self, Xtrain, Ftrain, Bounds, alpha=0.1, Acq_Fun = 'EI', epsilon= 0.1):
        self.Xtrain = Xtrain
        self.Ftrain = Ftrain
        self.Bounds = Bounds
        self.Acq_Fun = Acq_Fun
        self.alpha = alpha
        self.epsilon = epsilon

    def Global_Max(self, nseeds=20):
        
        PDist=pdist(self.Xtrain)
        Lmin=np.array([2*PDist.min(),0.5]).max()
        Lmax=PDist.max()
        # Gaussian Process
        kernel = RBF(length_scale_bounds=(Lmin,Lmax/2)) #length_scale=5
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=self.alpha, normalize_y=True)# n_restarts_optimizer=9
        gp.fit(self.Xtrain,self.Ftrain)
        
        ndim= self.Xtrain.shape[1]
        bds=np.array(self.Bounds)
        muNew=0
        x_seeds = np.random.uniform(0, 1,size=(nseeds, ndim))
        fMax=max(self.Ftrain)
        t=self.Xtrain.shape[0]
        
        if self.Acq_Fun == 'EI':
            fun=Acquisition_Functions.EI
            args=(gp,ndim,fMax)
        elif self.Acq_Fun =='PI':
             fun=Acquisition_Functions.PI
             args=(gp,ndim,fMax)
        elif self.Acq_Fun =='UCB2':
             fun=Acquisition_Functions.UCB2
             args = (gp, ndim, t)
            
        for dim in [0,1,ndim-1]:
            x_seeds[:,dim]= x_seeds[:,dim]*(bds[dim,1]-bds[dim,0])+bds[dim,0]
        
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res =minimize(fun,x0=x_try,args=args,method="L-BFGS-B",bounds=bds)
            
    #         Store it if better than previous minimum(maximum).
            if muNew is None or  -res.fun[0] >= muNew:
                x_new = res.x
                muNew = -res.fun[0]          
        return x_new, muNew
    

#########################################################################
#        Create Random Quadratic Function
#   Solving with Bayesian Optimization
#import matplotlib.pyplot as plt
#        
#Nfunc = 50 # number of functions generated
#Nweight = 6 # number of coeficients for the function
#Npoint = 11 ##numero de puntos para metodos sequenciales nuevos +1
#Fmax_BO=np.zeros((Npoint,Nfunc))
#
#np.random.seed(42)
#
#W=np.random.rand(Nfunc,Nweight)*100
#F_max=np.zeros((Nfunc,1))
#X_max=np.zeros((Nfunc,2))
#F_min=np.zeros((Nfunc,1))
#X_min=np.zeros((Nfunc,2))
#
#for i in range(Nfunc): 
#    # generate random cuadratic functions 
#    w=W[i,:]
#    def fun(X):
#       H = np.transpose([X[0]**2,X[1]**2,X[0]*X[1],X[0],X[1],1])
#       return -np.dot(H,w) # el menos es para que sean maximos
#    
#    bounds =[(-1,1),(-1,1)]
#    X_max[i],F_max[i] = Global_Maxima(fun,bounds)
#    X_min[i],F_min[i] = Global_Minima(fun,bounds)
#    
#    def func(X): ## Funcion escalada entre 0 y 1
#        return (fun(X)-F_min[i])/(F_max[i]-F_min[i])
#    
#    #   Bayesian Optimization 
#    X_init = np.array([[-1,-1], [1,1],[1,-1],[-1,1],[0,0]])
#    X_new=np.array([0,0])
#    
#    Y_init=np.zeros(X_init.shape[0])
#    for point in range(X_init.shape[0]):
#       Y_init[point] = func(X_init[point,:])
#    Y_new= func(X_new)
#   
#    for itr in range(0,Npoint):
#        X_init= np.vstack((X_init, X_new))
#        Y_init= np.hstack((Y_init, Y_new))
#        BO=Bay_Opt(X_init,Y_init,bounds,Acq_Fun = 'EI', epsilon= 0.1)
#        Fmax_BO[itr,i]=max(Y_init)
#        X_new, muNew=BO.Global_Max()
#        Y_new=func(X_new)
#
#plt.plot(Fmax_BO,'-o');
##
