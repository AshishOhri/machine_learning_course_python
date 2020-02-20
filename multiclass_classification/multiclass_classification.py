'''
Multiclass classification in the MNIST dataset to predict digits from 0 to 9
'''
# importing libraries
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
from scipy.optimize import fmin_cg
#importing data
f=loadmat('ex3data1.mat')
X=f['X']
y=f['y']
plt.rcParams['image.cmap']='gray' # to show the plot as grayscale
print("MNIST Data")
fig,ax=plt.subplots(10,10)
print('Shape of X: ',X.shape,'\nShape of y: ',y.shape)
for i in range(10):
    for j in range(10):
        integer=random.randint(0,4999)
        ax[i,j].imshow(X[integer].reshape(20,20).T)
        ax[i,j].axis('off')
plt.show()
input("Press enter to continue..")

def hyp_fn(theta,X):
    result=X@theta
    return 1/(1+np.exp(result)) # sigmoid function
def cost_fn(theta,X,y,lamb=0):
    theta=theta.reshape(-1,1)
    h=hyp_fn(theta,X)
    eps=np.finfo(float).eps # to deal with log(0)
    J=((-1/m)*np.sum(y.T@np.log(h+eps)+(1-y).T@(np.log(1-h+eps))))+(lamb/(2*m))*np.sum(np.sum(theta.T@theta))
    return J                                                        # regularization
def grad_fn(theta,X,y,lamb=0):
    theta=theta.reshape(-1,1)
    h=hyp_fn(theta,X)
    return((1/m)*(X.T@(h-y)))+((lamb/m)*(theta)).reshape(-1)
z=np.ones(X.shape[0]).reshape(-1,1)
X=np.append(z,X,axis=1)
m,n=X.shape
lamb=0.1
num_labels=10 # 10 digits to predict
theta=np.zeros((n,num_labels)) # 10 set of 401 X 1 weights exist for each digit
print('Initial Cost: ',cost_fn(theta[:,0],X,y,lamb))
input('Press enter to continue..')
for i in range(num_labels):
    theta[:,i]=fmin_cg(cost_fn,x0=theta[:,i],args=(X,np.array(y==(i+1),dtype=int),lamb),maxiter=50)
    # weights for each digit                                 # making y=1 for digit i+1 else 0 (Note 10 denotes digit 0) 
    print('Iteration {}/{}'.format(i+1,num_labels))           

print('Accuracy: ',np.mean((np.array([np.argmax(i)+1 for i  in (hyp_fn(theta,X))]).reshape(-1,1)==y))*100) # accuracy (in percentage)
