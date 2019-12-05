'''

Linear Regression on

'''

# Importing files
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# getting the data
f=open('ex1data1.txt','r')
data=f.read()
L=data.split('\n')
del(L[-1])
#print(L)
X=[i.split(',')[0] for i in L]
y=[i.split(',')[1] for i in L]
X=np.array(X,dtype='float32') # shape: 97
y=np.array(y,dtype='float32') # shape: 97
X=np.array([X]) # shape: 1X97
y=np.array([y]) # shape: 1X97
plt.plot(X,y,'rx') # plotting the data
plt.show()
n,m=X.shape
# Note: here m=97 i.e. there are 97 training examples withh 97 outputs and 1 feature
z=np.ones((m,1)).T # shape: 1X97
X=np.concatenate((z,X),axis=0)
#X=np.array([X]) # shape: 2X97
n,m=X.shape
print(X.shape,y.shape)
 #plot the x y values
f.close()

def hyp_fn(theta,X):
    return theta.T@X
def cost_fn(theta,X,m,y):
    h=hyp_fn(theta,X)
    return 1/(2*m)*np.sum((h-y)@(h-y).T)
def grad_fn(theta,X,m,y,alpha=0.01):
    h=hyp_fn(theta,X)
    return alpha/m*(X@(h-y).T)

theta=np.zeros((n,1))
#print(m)
#input('Press enter to continue..')
print('Initial Cost: ',cost_fn(theta,X,m,y))
#print(theta.shape)
#print(theta.T.shape,X.shape,hyp_fn(theta,X).shape)

input('Press enter to continue..')
for i in range(1500):
    theta-=grad_fn(theta,X,m,y)
    print("Cost/Error: ",cost_fn(theta,X,m,y))
print("Theta: ",theta)
input('Press enter to continue..')
print(hyp_fn(theta,X).shape)
plt.plot(X[[1],:][0],y[0],'rx',label='Training data')
plt.plot(X[[1],:][0],hyp_fn(theta,X)[0],label='predicted values(linear regression)')
plt.legend()
plt.show()


input('Press enter to continue..')
sx=np.linspace(-10,10,num=100)
sy=np.linspace(-1,4,num=100)
sz=np.zeros((sx.shape[0],sy.shape[0]))
for i in  range(sx.shape[0]):
    for j in range(sy.shape[0]):
        t=np.array([sx[j],sy[i]])
        sz[i][j]=cost_fn(t,X,m,y)
sx,sy=np.meshgrid(sx,sy)
sz=np.array(np.array(sz))
print(sx.shape,sy.shape,sz.shape)
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot_surface(sx,sy,sz)
plt.show()

plt.figure(figsize=(10,5))
cp = plt.contour(sx, sy, sz)
plt.clabel(cp, inline=1, fontsize=10)
plt.show()