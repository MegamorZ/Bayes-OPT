# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:45:09 2020

Metodo simplex explicado en:
M.A.Bezerra etal. / MicrochemicalJournal124 (2016) 45 –54

@author: megam
"""
import numpy as np
import pandas as pd

######
#def Z(R):
#    X=R[0]
#    Y=R[1]
#    #Z=R[2]
#    #X0=0
#    #Y0=0
#    return (1-(X**2+Y**3))* np.exp(-(X**2+Y**2)/2)#+0*Z #-2*(X+X0)**2-4*(Y+Y0)**2 -3*(X+X0)-5*(Y+Y0)-0.05*(X+X0)*(Y+Y0) +1000 
######
    
np.random.seed(46)

# Simplex initial Vertex

#SIV =np.array ([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
#               [1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
#               [0.5,	0.87,	 0,	0,	0,	0,	0,	0,	0,	0],
#               [0.5,	0.29,	0.82,	0,	0,	0,	0,	0,	0,	0],
#               [0.5,	0.29,	0.2,	0.79,	0,	0,	0,	0,	0,	0],
#               [0.5,	0.29,	0.2,	0.16,	0.78,	0,	0,	0,	0,	0],
#               [0.5,	0.29,	0.2,	0.16,	0.13,	0.76,	0,	0,	0,	0],
#               [0.5,	0.29,	0.2,	0.16,	0.13,	0.11,	0.76,	0,	0,	0],
#               [0.5,	0.29,	0.2,	0.16,	0.13,	0.11,	0.094	,0.75,	0,	0],
#               [0.5,	0.29,	0.2,	0.16,	0.13,	0.11,	0.094,	0.083,	0.75,	0]])



class Simplex_Opt:
    def __init__(self, X0, Step=1):
        self.Step = Step #step
        self.X0 = X0 #starting point
        self.dim=len(X0) # dimension
        self.SIV = np.array ([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
               [1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
               [0.5,	0.87,	 0,	0,	0,	0,	0,	0,	0,	0],
               [0.5,	0.29,	0.82,	0,	0,	0,	0,	0,	0,	0],
               [0.5,	0.29,	0.2,	0.79,	0,	0,	0,	0,	0,	0],
               [0.5,	0.29,	0.2,	0.16,	0.78,	0,	0,	0,	0,	0],
               [0.5,	0.29,	0.2,	0.16,	0.13,	0.76,	0,	0,	0,	0],
               [0.5,	0.29,	0.2,	0.16,	0.13,	0.11,	0.76,	0,	0,	0],
               [0.5,	0.29,	0.2,	0.16,	0.13,	0.11,	0.094	,0.75,	0,	0],
               [0.5,	0.29,	0.2,	0.16,	0.13,	0.11,	0.094,	0.083,	0.75,	0]])

    def Initial_Points(self):
        # Initial Points
        Xrep=np.array([self.X0,]*((self.dim)+1))
        P=Xrep + self.SIV[0:(self.dim)+1,0:self.dim]*self.Step
        return P

    def Simplex(self,X_init):
        #X_init is dtatframe array X and Responses
        #df = pd.DataFrame(X_init)
        X_init.sort_values(X_init.columns[-1],ascending=False, inplace=True)#sort by response
        M=X_init.iloc[0:(self.dim),0:(self.dim)].mean(axis=0)
        R =  2*M - X_init.iloc[self.dim,0:(self.dim)]
        
        return R    #return R without response  
  
    def Simplex_Mod(self,X_init,fun):
         #X_init is dtatframe array X and Responses
        X_init.sort_values(X_init.columns[-1],ascending=False, inplace=True)#sort by response
        M=X_init.iloc[0:(self.dim),0:(self.dim)].mean(axis=0)
        R =  2*M - X_init.iloc[self.dim,0:(self.dim)]
        E =  3*M - 2*X_init.iloc[self.dim,0:(self.dim)] 
        # conditions for expansion and contraction
        alpha = 1
        if fun(R) > X_init.iloc[0,-1]:
            if fun(E) >= fun(R):
                alpha = 2
            else:
                alpha = 1
        elif fun(R) < X_init.iloc[1,-1] and fun(R) > X_init.iloc[2,-1]:
            alpha = 0.5
        elif fun(R) < X_init.iloc[2,-1]:
            alpha = -0.5
            
        R = (alpha+1)*M - alpha*X_init.iloc[2,:-1]
        R.at[self.dim]= fun(R)
        
        return R    #return R with response
        
        
############################################
##modified simplex
#X0=[-3,-3]        
#Simplx=Simplex_Opt(X0,Step=1)  
#Pinit=Simplx.Initial_Points()
#df=pd.DataFrame(np.hstack((Pinit, Z(np.transpose(Pinit)).reshape(-1,1))))
#dfaux=df.copy()
#
#niter=50
#for i in range(1,niter):  
#    df.sort_index(inplace=True)  
#    dfaux.sort_values(dfaux.columns[-1],ascending=False, inplace=True)  
#    R=Simplx.Simplex_Mod(dfaux,Z)
#    df.loc[i+2]=R
#    dfaux.iloc[len(X0)]=R
#
#
#import matplotlib.pyplot as plt
##df.sort_values(dfaux.columns[-1],ascending=True, inplace=True)
#plt.plot(df[2])
#    
####################################
#Simplex         
#X0=[-3,-3]        
#Simplx=Simplex_Opt(X0,Step=1)  
#Pinit=Simplx.Initial_Points()
#df=pd.DataFrame(np.hstack((Pinit, Z(np.transpose(Pinit)).reshape(-1,1))))
#dfaux=df.copy()
#
#
#niter=50
#for i in range(1,niter):
#    df.sort_index(inplace=True)
#    dfaux.sort_values(dfaux.columns[-1],ascending=False, inplace=True)
#    R=Simplx.Simplex(dfaux)
#    R.at[len(X0)]= Z(R)
#    df.loc[i+2]=R
#    dfaux.iloc[len(X0)]=R
#    
#
#df.sort_index(inplace=True)
#
#import matplotlib.pyplot as plt
##df.sort_values(dfaux.columns[-1],ascending=True, inplace=True)
#plt.plot(df[2])
##########################################
