'''

 linear regression with one variable to predict profits for a food truck as per
 population in different cities

'''

# Importing files
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# getting the data
f=open('ex1data1.txt','r')
data=f.read()
f.close()
L=data.split('\n')
del(L[-1]) # Empty value

X=[i.split(',')[0] for i in L]
y=[i.split(',')[1] for i in L]
X=np.array(X,dtype='float32') # shape: 97
y=np.array(y,dtype='float32') # shape: 97
X=np.array([X]) # shape: 1X97
y=np.array([y]) # shape: 1X97
print('Scatter Plot')
plt.title('Scatter plot of training plot')
plt.xlabel('Population of city in 10000s')
plt.ylabel('Profit in $10000s')
plt.plot(X,y,'rx') # plotting the data
plt.show()
n,m=X.shape
# Note: here m=97 i.e. there are 97 training examples withh 97 outputs and 1 feature
z=np.ones((m,1)).T # shape: 1X97
X=np.concatenate((z,X),axis=0)
n,m=X.shape

def hyp_fn(theta,X):
    return theta.T@X

def cost_fn(theta,X,m,y):
    h=hyp_fn(theta,X)
    return 1/(2*m)*np.sum((h-y)@(h-y).T)

def grad_fn(theta,X,m,y,alpha=0.01):
    h=hyp_fn(theta,X)
    return alpha/m*(X@(h-y).T)

theta=np.zeros((n,1))
print('Initial Cost: ',cost_fn(theta,X,m,y))

input('Press enter to continue..')
for i in range(1500):
    theta-=grad_fn(theta,X,m,y)
    print("Cost/Error: ",cost_fn(theta,X,m,y))
print("Theta: ",theta)

input('Press enter to continue..')
print('Regression plot')
plt.title('Line of fit in regression plot')
plt.xlabel('Population of city in 10000s')
plt.ylabel('Profit in $10000s')    
plt.plot(X[[1],:][0],y[0],'rx',label='Training data')
plt.plot(X[[1],:][0],hyp_fn(theta,X)[0],label='predicted values(linear regression)')
plt.legend()
plt.show()


input('Press enter to continue..')
print('Surface plot')
sx=np.linspace(-10,10,num=100) # linearly spaced 100 points b/w -10 and 10
sy=np.linspace(-1,4,num=100)
sz=np.zeros((sx.shape[0],sy.shape[0]))
minv=np.inf # to get minimum cost value to plot on graph
minsx,minsy=0,0 # to get sx(theta(1)) and sy(theta(1)) value at minimum cost value
for i in  range(sx.shape[0]):
    for j in range(sy.shape[0]):
        t=np.array([[sx[i]],[sy[j]]])
        cost=cost_fn(t,X,m,y)
        sz[i][j]=cost
        if cost<minv:
            minv=cost
            minsx=sx[i]
            minsy=sy[j]
sx,sy=np.meshgrid(sx,sy)
sz=np.array(np.array(sz)).T

ax=plt.axes(projection='3d')
plt.title('Cost values wrt theta values')
plt.xlabel('theta(0)')
plt.ylabel('theta(1)')
ax.plot_surface(sx,sy,sz)
ax.view_init(elev=30, azim=-120) # to set the graph viewing position
plt.show()

input('Press enter to continue..')
print('Contour plot')
plt.title('Cost values wrt theta values')
plt.xlabel('theta(0)')
plt.ylabel('theta(1)')
cp = plt.contour(sx, sy, sz, np.logspace(-2, 3, 20))
plt.clabel(cp, inline=1, fontsize=10)
plt.plot(minsx,minsy,marker='x',color='r')
plt.show()