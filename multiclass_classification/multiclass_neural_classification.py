'''
Multiclass classification in the MNIST dataset to predict digits from 0 to 9
'''
#importing libraries
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
#importing data
f=loadmat('ex3data1.mat')
X=f['X']
y=f['y']
plt.rcParams['image.cmap']='gray' # to show plot as grayscale
print("MNIST Data")
print('Shape of X: ',X.shape,'\nShape of y: ',y.shape)
fig,ax=plt.subplots(10,10)
for i in range(10): # 10 by 10 randomized digit plot
    for j in range(10):
        integer=random.randint(0,4999)
        ax[i,j].imshow(X[integer].reshape(20,20).T)
        ax[i,j].axis('off')
plt.show()
input("Press enter to continue..")
f=loadmat('ex3weights.mat') # pretrained weights
theta1=f['Theta1']
theta2=f['Theta2']
print('Theta1 Shape: ',theta1.shape,'\nTheta2 Shape: ',theta2.shape)
def sigmoid(val):
    return 1/(1+np.exp(-val))
def predict(x): # Neural Network
    z=np.ones(x.shape[0]).reshape(-1,1)
    x=np.append(z,x,axis=1)
    res=x@theta1.T
    res=sigmoid(res)
    z=np.ones(res.shape[0]).reshape(-1,1)
    res=np.append(z,res,axis=1)
    res=res@theta2.T
    return np.array([np.argmax(i)+1 for i in res]).reshape(-1,1)
print('Accuracy:', np.mean(predict(X)==y)*100) # Accuracy
input('Press enter to continue..')
while True: # randomized predictions and digit plot comparison
    integer=random.randint(0,4999)
    print('Prediction: ',predict(X[integer].reshape(1,-1))[0][0]%10)
    plt.imshow(X[integer].reshape(20,20).T)
    plt.axis('off')
    plt.show()
    if input('Press q to exit or any other key to continue ')=='q':
        break
