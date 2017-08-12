# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:51:12 2017

@author: leo
"""

import numpy as np
from scipy.stats import chi2

def ellipsoid_confidence(cov, mean=(0,0,0), prob=0.95, N=100):
    """ Returns 3D points defining an ellipsoid at a given confidence.
    
    Parameters
    -----------
    cov: array 3D (3,3)
        Covariance
    prob: float
        Confidence interval
    N: int
        Number of points returned
    
    Returns
    ------------
    Three 2D arrays (N, N) defining the ellipse
    """        
    
    # the confidance is ruled by the chi-square with degree 2
    chi2_3 = chi2.ppf(prob, 3)
    
    # U defines the rotation of the ellipse whereas
    # S gives us the eignevalues defining the amplitude of the
    # ellipse in the two dimensions
    U,S,_ = np.linalg.svd(cov)  
    
    # the ellipsoid is defined in spherical coordinates
    a, b, c = np.sqrt(S * chi2_3)
    u = np.linspace(0, 2*np.pi, N)
    v = np.linspace(0, np.pi, N)
    
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    
    D = np.array([x.ravel(), y.ravel(), z.ravel()])
    
    # applying the rotation
    D = np.dot(U, D).T
    
    # moving the data to the given mean
    X = D[:,0].reshape(N,N) + mean[0]
    Y = D[:,1].reshape(N,N) + mean[1]
    Z = D[:,2].reshape(N,N) + mean[2]
    
    return X, Y, Z
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    # generate some 2d standard isotropic gaussian data
    X = np.random.randn(100, 3)
    
    MEAN = (10, 5, 3)
    COV = np.array([[10, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    # tranform the data based on the covariance we want                
    U,S,V = np.linalg.svd(COV)  
    T = np.dot(U, np.sqrt(np.diag(S))) 
    Xt = np.dot(T, X.T).T
    Xt[:,0] += MEAN[0]
    Xt[:,1] += MEAN[1]           
    Xt[:,2] += MEAN[2]  
    
    # check if correct
    print(np.cov(Xt.T))    

    # plot
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xt[:,0], Xt[:,1], Xt[:,2], color='r')

    X, Y, Z = ellipsoid_confidence(COV, mean=MEAN, prob=0.95, N=100)   
    
    #ax.plot_surface(X, Y, Z,  rstride=4, cstride=4, color='k')
    ax.plot_wireframe(X, Y, Z,  rstride=4, cstride=4, color='k')
    
    plt.show()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')