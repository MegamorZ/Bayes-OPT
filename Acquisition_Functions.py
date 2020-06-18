# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:33:51 2020

Funciones de Adquisicion para optimizacion bayesiana.

@author: megam
"""
import numpy as np
import scipy
    
def UCB(x, gp, ndim):
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    x1=np.array(x).reshape(-1,ndim)
    y_pred, sigma = gp.predict(x1, return_std=True)
    return -(y_pred + 1.96 * sigma)

def PI(x, gp, ndim,fMax, epsilon=0.1):
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
	#epsilon = 0.1
	x1=np.array(x).reshape(-1,ndim)
	muNew, stdNew = gp.predict(x1, return_std=True)
	#fMax=max(Y_init)
    
	Z = (muNew - fMax - epsilon)/stdNew

	return -scipy.stats.norm.cdf(Z) 

def EI(x,  gp, ndim,fMax, epsilon=0.1):
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
	#epsilon = 0.1
	x1=np.array(x).reshape(-1,ndim)
	muNew, stdNew = gp.predict(x1, return_std=True)
	#fMax=max(Y_init)
	Z = (muNew - fMax - epsilon)/stdNew
	return -((muNew - fMax - epsilon)* scipy.stats.norm.cdf(Z) + stdNew*scipy.stats.norm.pdf(Z))


def UCB2(x, gp, ndim, t,delta = 0.1,v=1):
	"""
	Upper confidence bound acquisition function
	INPUT:
		- muNew: predicted mean
		- stdNew: sigma (square root of variance) of predicted point in grid
		- t: number of iteration (t=Xtrain.shape[0])
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
	#t=X_init.shape[0]
#	v=3
#	delta=0.1
	x1=np.array(x).reshape(-1,ndim)
	muNew, stdNew = gp.predict(x1, return_std=True)
	#fMax=max(Y_init)
	#Kappa = np.sqrt( v* (2*  np.log((t**(d/2. + 2))*(np.pi**2)/(3. * delta)  )))
	Kappa = delta*((v**d)/t) 
	#plt.plot(t,Kappa,'o')
	return -(muNew + Kappa * stdNew)