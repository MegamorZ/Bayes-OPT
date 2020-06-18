# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:42:42 2020

Bayesian Optimization con sklearn. en nD
visualizacion en 2D - for animation

@author: megamorz
"""

#import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import scipy
from scipy.optimize import minimize
#import random
from matplotlib import pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

np.random.seed(46)

#optimization function
def z_func(x,y):
    return   (1-(x**2+y**3))*np.exp(-(x**2+y**2)/2)


# bounds poner a mano por cada variable (min, max)
bounds=[(-3,3),(-3,3)]

##initial values
X_init = np.array([[-2.5,-2.5], [2.5,2.5],[2.5,-2.5],[-2.5,2.5]])
x_EI_max=[(0,0)]
Npoint=30 #numero de puntos nuevos +1

F_EI= np.ones(Npoint-1)
F_PI= np.ones(Npoint-1)
F_max= np.ones(Npoint-1)

for iterati in range(1,Npoint):
    
    X_init= np.vstack((X_init, x_EI_max))
    Y_init = z_func(X_init[:,0],X_init[:,1])
    ndim=X_init.shape[1]
    
    #Core Gaussian process con sklearn
    
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gp = GaussianProcessRegressor(kernel=kernel,alpha=0.1,n_restarts_optimizer=9, normalize_y=False)
    gp.fit(X_init, Y_init)
    
    ###############################################################################
    #acquisition functions
    
    def UCB(x):
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        x1=np.array(x).reshape(-1,ndim)
        y_pred, sigma = gp.predict(x1, return_std=True)
        return -(y_pred + 1.96 * sigma)
    
    def PI(x):
    	"""
    	Probability of improvement acquisition function
    	INPUT:
    		- muNew: mean of predicted point in grid
    		- stdNew: sigma (square root of variance) of predicted point in grid
    		- fMax: observed or predicted maximum value (depending on noise p.19 [Brochu et al. 2010])
    		- epsilon: trade-off parameter (>=0)
    	OUTPUT:
    		- PI: probability of improvement for candidate point
    	As describend in:
    		E Brochu, VM Cora, & N de Freitas (2010): 
    		A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning,
    		arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
    	"""
    	epsilon = 0.1
    	x1=np.array(x).reshape(-1,ndim)
    	muNew, stdNew = gp.predict(x1, return_std=True)
    	fMax=max(Y_init)
        
    	Z = (muNew - fMax - epsilon)/stdNew
    
    	return -scipy.stats.norm.cdf(Z) 
    
    def EI(x):
    	"""
    	Expected improvement acquisition function
    	INPUT:
    		- muNew: mean of predicted point in grid
    		- stdNew: sigma (square root of variance) of predicted point in grid
    		- fMax: observed or predicted maximum value (depending on noise p.19 Brochu et al. 2010)
    		- epsilon: trade-off parameter (>=0) 
    			[Lizotte 2008] suggest setting epsilon = 0.01 (scaled by the signal variance if necessary)  (p.14 [Brochu et al. 2010])		
    	OUTPUT:
    		- EI: expected improvement for candidate point
    	As describend in:
    		E Brochu, VM Cora, & N de Freitas (2010): 
    		A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning, 
    		arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
    	"""
    	epsilon = 0.1
    	x1=np.array(x).reshape(-1,ndim)
    	muNew, stdNew = gp.predict(x1, return_std=True)
    	fMax=max(Y_init)
    	Z = (muNew - fMax - epsilon)/stdNew
    	return -((muNew - fMax - epsilon)* scipy.stats.norm.cdf(Z) + stdNew*scipy.stats.norm.pdf(Z))
    
    
    def UCB2(x, t=X_init.shape[0]):
    	"""
    	Upper confidence bound acquisition function
    	INPUT:
    		- muNew: predicted mean
    		- stdNew: sigma (square root of variance) of predicted point in grid
    		- t: number of iteration
    		- d: dimension of optimization space
    		- v: hyperparameter v = 1*
    		- delta: small constant (prob of regret)
    		*These bounds hold for reasonably smooth kernel functions.
    		[Srinivas et al., 2010]
    		OUTPUT:
    		- UCB: upper confidence bound for candidate point
    	As describend in:
    		E Brochu, VM Cora, & N de Freitas (2010): 
    		A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning, 
    		arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
    	"""
    	d=ndim
    	t=X_init.shape[0]
    	v=3
    	delta=0.1
    	x1=np.array(x).reshape(-1,ndim)
    	muNew, stdNew = gp.predict(x1, return_std=True)
    	#fMax=max(Y_init)
    	#Kappa = np.sqrt( v* (2*  np.log((t**(d/2. + 2))*(np.pi**2)/(3. * delta)  )))
    	Kappa = delta*((v**d)/t) 
    	#plt.plot(t,Kappa,'o')
    	return -(muNew + Kappa * stdNew)
    ###############################################################################
    #Maximization utility function
    
    n_iter=10
    y_max=max(Y_init)
    #max_acq=y_max
    fPI=0
    fEI=0
    fUCB=0
    fUCB2=0
    
    # preparar las seeds
    bds=np.array(bounds)
    
    #ndim=bds.shape[0]
    xe = np.random.uniform(0, 1,size=(n_iter, ndim))
    x_seeds= xe
    #print(ndim)
    for dim in [0,1,ndim-1]:
            x_seeds[:,dim]= xe[:,dim]*(bds[dim,1]-bds[dim,0])+bds[dim,0]
    for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
     #       print(x_try)
            res_PI=minimize(fun=PI,x0=x_try,method="L-BFGS-B",bounds=bds)
            res_EI=minimize(fun=EI,x0=x_try,method="L-BFGS-B",bounds=bds)
            res_UCB=minimize(fun=UCB,x0=x_try,method="L-BFGS-B",bounds=bds)
            res_UCB2=minimize(fun=UCB2,x0=x_try,method="L-BFGS-B",bounds=bds)
            
    #         Store it if better than previous minimum(maximum).
            if fPI is None or  -res_PI.fun[0] >= fPI:
                x_PI_max = res_PI.x
                fPI = -res_PI.fun[0]
            if fEI is None or -res_EI.fun[0] >= fEI:  
                x_EI_max = res_EI.x
                fEI = -res_EI.fun[0]
            if fUCB is None or -res_UCB.fun[0] >= fUCB:   
                x_UCB_max = res_UCB.x        
                fUCB = -res_UCB.fun[0]
            if fUCB2 is None or -res_UCB2.fun[0] >= fUCB2:  
                x_UCB2_max = res_UCB2.x
                fUCB2 = -res_UCB2.fun[0]
                
    F_EI[iterati-1]=fEI
    F_PI[iterati-1]=fPI 
    F_max[iterati-1]=y_max
    print(y_max)
#    print ("X PI:", "PI:",sep="---")
#    print (x_PI_max,fPI,sep="---") 
#    print ("X EI:", "EI:",sep="---")
#    print (x_EI_max,fEI,sep="---")
#    print ("X UCB:", "UCB:",sep="---")
#    print (x_UCB_max,fUCB,sep="---")
#    print ("X UCB2:", "UCB2:",sep="---")
#    print (x_UCB2_max,fUCB2,sep="---")
    ################################################
#    #plotting
#    
#    #generate prediction array
#    x = np.arange(-3.0,3.0,0.1)
#    y = np.arange(-3.0,3.0,0.1)
#    X,Y = meshgrid(x, y) # grid of point
#    Z = z_func(X, Y)
#    
#    Xr=X.reshape(-1,1)
#    Yr=Y.reshape(-1,1)
#    # z=Z.reshape(-1,1)
#    Xar=np.append(Xr,Yr,axis=1)
#    
#    y_pred, sigma = gp.predict(Xar, return_std=True)
#    y_pred=y_pred.reshape(X.shape)
#    sigma=sigma.reshape(X.shape)
#    
#    # Plot
#    Npoints=X_init.shape[0]
#    plt.figure(figsize=(16, 14), dpi= 80)
#    plt.suptitle('r =UCB2, g= PI, y= EI, Num points = %1.0f' %Npoints)
#    plt.subplot(2,3,2)
#    im = imshow(y_pred,cmap=cm.RdBu, extent=[-3,3,3,-3]) # drawing the function
#    # adding the Contour lines with labels
#    cset = contour(y_pred,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2, extent=[-3,3,-3,3])
#    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
#    colorbar(im) # adding the colobar on the right
#    plt.plot(X_init[:,0],X_init[:,1],'o',color='black')
#    plt.plot(x_PI_max[0],x_PI_max[1],'*',markersize=20, color='green')
#    plt.plot(x_EI_max[0],x_EI_max[1],'*',markersize=20, color='yellow')
#    plt.plot(x_UCB2_max[0],x_UCB2_max[1],'*',markersize=20, color='red')
#    title('Predicted surface')
#    
#    plt.subplot(2,3,1)
#    im = imshow(Z,cmap=cm.RdBu, extent=[-3,3,3,-3]) # drawing the function
#    # adding the Contour lines with labels
#    cset = contour(Z,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2, extent=[-3,3,-3,3])
#    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
#    colorbar(im) # adding the colobar on the right
#    plt.plot(X_init[:,0],X_init[:,1],'o',color='black')
#    plt.plot(x_PI_max[0],x_PI_max[1],'*',markersize=20, color='green')
#    plt.plot(x_EI_max[0],x_EI_max[1],'*',markersize=20, color='yellow')
#    plt.plot(x_UCB2_max[0],x_UCB2_max[1],'*',markersize=20, color='red')
#    title('Real Surface')
#    
#    plt.subplot(2,3,3)
#    im = imshow(-PI(Xar).reshape(X.shape),cmap=cm.RdBu, extent=[-3,3,3,-3]) # drawing the function
#    # adding the Contour lines with labels
#    cset = contour(-PI(Xar).reshape(X.shape) ,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2, extent=[-3,3,-3,3])
#    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
#    colorbar(im) # adding the colobar on the right
#    plt.plot(X_init[:,0],X_init[:,1],'o',color='black')
#    plt.plot(x_PI_max[0],x_PI_max[1],'*',markersize=20, color='green')
#    plt.plot(x_EI_max[0],x_EI_max[1],'*',markersize=20, color='yellow')
#    plt.plot(x_UCB2_max[0],x_UCB2_max[1],'*',markersize=20, color='red')
#    title('PI function')
#    
#    plt.subplot(2,3,4)
#    im = imshow(-EI(Xar).reshape(X.shape),cmap=cm.RdBu, extent=[-3,3,3,-3]) # drawing the function
#    # adding the Contour lines with labels
#    cset = contour(-EI(Xar).reshape(X.shape) ,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2, extent=[-3,3,-3,3])
#    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
#    colorbar(im) # adding the colobar on the right
#    plt.plot(X_init[:,0],X_init[:,1],'o',color='black')
#    plt.plot(x_PI_max[0],x_PI_max[1],'*',markersize=20, color='green')
#    plt.plot(x_EI_max[0],x_EI_max[1],'*',markersize=20, color='yellow')
#    plt.plot(x_UCB2_max[0],x_UCB2_max[1],'*',markersize=20, color='red')
#    title('EI function')
#    
#    plt.subplot(2,3,5)
#    im = imshow(-UCB2(Xar).reshape(X.shape),cmap=cm.RdBu, extent=[-3,3,3,-3]) # drawing the function
#    # adding the Contour lines with labels
#    cset = contour(-UCB2(Xar).reshape(X.shape) ,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2, extent=[-3,3,-3,3])
#    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
#    colorbar(im) # adding the colobar on the right
#    plt.plot(X_init[:,0],X_init[:,1],'o',color='black')
#    plt.plot(x_PI_max[0],x_PI_max[1],'*',markersize=20, color='green')
#    plt.plot(x_EI_max[0],x_EI_max[1],'*',markersize=20, color='yellow')
#    plt.plot(x_UCB2_max[0],x_UCB2_max[1],'*',markersize=20, color='red')
#    title('UCB2 function')
    #############################################################################
    #plot Ymax
    #plt.plot(iterati,y_max,'o')
    
    #Plot acq funcion value
    #plt.plot(iterati,fEI,'o')
    
    #############################################################################
    #save figures
#    savefile = 'USB21' + str(iterati) + '.png'   # file might need to be replaced by a string
#    plt.savefig(savefile)
#    
 ##############################################################   
## plot distance between points
    
N_punto = np.arange(1,X_init.shape[0]-2,1)

Dist=np.zeros(X_init.shape[0]-2)

#print(N_punto.shape)
for k in (N_punto):
          #print(k)
          Dist[k-1]=np.sqrt(np.sum((X_init[k,:]-X_init[k-1,:])**2))

#print(N_punto)    
#print(Dist)


#plt.plot(Dist,'-o')
#plt.plot(max(N_punto)+1,np.sqrt(np.sum((X_init[-1,:]-x_EI_max)**2)),'*')
#plt.plot(max(N_punto)+1,np.sqrt(np.sum((X_init[-1,:]-x_PI_max)**2)),'*',color='green')
#plt.plot(max(N_punto)+1,np.sqrt(np.sum((X_init[-1,:]-x_UCB2_max)**2)),'*',color='red')   
##################################################    

    
#Plot acq funcion value

plt.figure(figsize=(16, 14), dpi= 80)
 #  plt.suptitle('r =UCB2, g= PI, y= EI, Num points = %1.0f' %Npoints)
plt.subplot(2,1,1) 
#Plot acq funcion value  
plt.plot(F_EI,'-o') 
plt.plot(F_PI,'-o') 
 #plot Ymax
plt.subplot(2,1,2) 
plt.plot(F_max,'-o')   