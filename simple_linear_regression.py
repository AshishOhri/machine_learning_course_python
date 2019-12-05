'''

This notebook is a simple implementation of linear regression from scratch in python.

'''
# Equation y=theta(0)*x0+theta(1)*x1
# We are trying to get the values of theta which is where the learning in machine learning takes place.
import numpy as np
def hyp_fn(theta,X): # hypothesis function
    return theta.T@X
def cost_fn(theta,X,m,y): # cost function
    h=hyp_fn(theta,X)
    return 1/(2*m)*np.sum((h-y).T@(h-y))
def gradient_fn(theta,X,m,y,alpha=0.3): # calculates the gradient descent, changing the value of theta iteratively
    for i in range(100):
        h=hyp_fn(theta,X)
        theta-=alpha/m*(X@(h-y))
        print('Cost/Error:',cost_fn(theta,x,len(x),y))
    return theta
theta=np.array([[0,0]]).T # shape (before transpose): 1X2 (after transpose):2X1
y=np.array([[2]]) #shape: 1X1
x=np.array([[0],[1]]) # shape: 2X1
y=y.astype('float32')
x=x.astype('float32')
theta=theta.astype('float32')
theta=gradient_fn(theta,x,len(x),y)
print('\nTheta = ',theta)
print("y=%f*x0+%f*x1"%(theta[0],theta[1])) # We get the equation of theta as y=2x and rightly so
                                           # This can be cross checked by looking at the values of y and x array