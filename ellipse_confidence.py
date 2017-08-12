# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:23:57 2017

@author: leo
"""

import numpy as np
from scipy.stats import chi2

def ellipse_confidence(cov, mean=(0,0), prob=0.95, N=100):
    """ Returns 2D points defining an ellipse at a given confidence.
    
    Parameters
    -----------
    cov: array 2D (2,2)
        Covariance
    prob: float
        Confidence interval
    N: int
        Number of points retuned
    
    Returns
    ------------
    An array 2D (N, 2) defining the ellipse
    """        
    
    # the confidance is ruled by the chi-square with degree 2
    chi2_2 = chi2.ppf(prob, 2)
    
    # U defines the rotation of the ellipse whereas
    # S gives us the eignevalues defining the amplitude of the
    # ellipse in the two dimensions
    U,S,_ = np.linalg.svd(cov)  
    
    # the ellipse is defined in polar coordinates
    a, b = np.sqrt(S * chi2_2)
    theta = np.linspace(0, 2*np.pi, N)
    
    x = (a*b*np.cos(theta))/(np.sqrt(b**2*np.cos(theta)**2 + a**2*np.sin(theta)**2))
    y = (a*b*np.sin(theta))/(np.sqrt(b**2*np.cos(theta)**2 + a**2*np.sin(theta)**2))
    D = np.array([x,y])
    
    # applying the rotation
    D = np.dot(U, D).T
    
    # moving the data to the given mean
    D[:,0] += mean[0]
    D[:,1] += mean[1]
    
    return D

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # generate some 2d standard isotropic gaussian data
    X = np.random.randn(10000, 2)
    
    # tranform the data based on the covariance we want
    MEAN = (10,5)
    COV = np.array([[6, 2],
                    [2, 3]])
    U,S,V = np.linalg.svd(COV)  
    T = np.dot(U, np.sqrt(np.diag(S))) 
    Xt = np.dot(T, X.T).T
    Xt[:,0] += MEAN[0]
    Xt[:,1] += MEAN[1]
    
    # check if correct
    print(np.cov(Xt.T))
    
    # plot
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(X[:,0], X[:,1], 'b.')
    plt.axis('equal')
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(Xt[:,0], Xt[:,1], 'r.')
    plt.axis('equal')
    plt.grid()
    
    Dt = ellipse_confidence(COV, MEAN, prob=0.95)
    
    plt.plot(Dt[:,0], Dt[:,1], 'b', linewidth=2)