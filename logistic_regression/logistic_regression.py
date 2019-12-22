'''

 logistic regression to predict whether student gets admitted to
 a university or not based on two exams data

'''

# Importing files
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# getting the data
f=open('ex2data1.txt','r')
data=f.read()
f.close()
L=data.split('\n')
del(L[-1]) # Empty value

plt_array=np.array([i.split(',') for i in L],dtype='float32')
print('Scatter Plot')
plt_array_1=np.array([plt_array[i] for i in range(len(plt_array.T[2])) if plt_array.T[2][i]==1])
plt_array_2=np.array([plt_array[i] for i in range(len(plt_array.T[2])) if plt_array.T[2][i]==0])

plt.figure(figsize=(10,7))
plt.title('Scatter plot of training plot')
plt.xlabel('Marks in the first exam')
plt.ylabel('Marks in the second exam')
plt.scatter(plt_array_1.T[0],plt_array_1.T[1],marker='+',s=85,label='Admitted') # plotting the data
plt.scatter(plt_array_2.T[0],plt_array_2.T[1],marker='o',c='yellow',edgecolors='black',s=85,label='Not Admitted')
plt.legend(loc='upper right')
plt.show()


X=[i.split(',')[:2:] for i in L]
y=[i.split(',')[2] for i in L]

X=np.array(X,dtype='float32') # shape: 100X2
y=np.array(y,dtype='float32') # shape: 100
y=np.array([y]).T # shape: 100X1
m,n=X.shape
print(m,n)
# Note: here m=100 and n=2 i.e. there are 100 training examples with 100 outputs and 2 feature
z=np.ones((m,1)) # shape:100X1

X=np.concatenate((z.T,X.T),axis=0).T # shape: 100X3

m,n=X.shape

def hyp_fn(theta,X):
    x=X@theta
    return (1/(1+np.exp(-x)))
def cost_fn(theta,X,y):
    h=hyp_fn(theta,X)
    eps=np.finfo(float).eps # tdeal with log(0)
    #print(eps)
    return ((-1/m)*np.sum(y.T@np.log(h+eps)+(1-y).T@(np.log(1-h+eps))))
def grad_fn(theta,X,y):
    h=hyp_fn(theta,X)
    return((0.001/m)*(X.T@(h-y)))

theta=np.zeros((n,1))
print('Initial Cost: ',cost_fn(theta,X,y))
array=[]
input('Press enter to continue..')
iterations=500000
for i in range(iterations):
    array.append(cost_fn(theta,X,y))
    print("Cost/Error: ",cost_fn(theta,X,y))
    theta-=grad_fn(theta,X,y)
print("Theta: ",theta)
plt.plot([i for i in range(1,iterations+1)],array)
plt.show()