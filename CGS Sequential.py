# Disable numpy threading to get useful timings
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Import Libraries
import numpy as np
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy import sin
from numpy import cos
import seaborn as sns

# Define discretizing function
def f(mu, x):
    result = np.sin(10*(mu+x))/(np.cos(100*(mu-x))+1.1)
    return result

# Start evaluating run-time
wt = time.time()

# Matrix definition
# Case 1:
'''
m=3000
n=300
W=np.zeros((m,n))
for i in range(m):
        for j in range(n):
            W[i, j] = f(i, j)
'''
# Case 2:
'''
m=3000
n=300
W=np.arange(1, m*n+1, 1, dtype='d')
W = np.reshape(W,(m,n))
W = W+np.eye(m,n)
'''
# Case 3:
'''
data = loadmat('full_matrix.mat')
W = data['good_cond']
m = W.shape[0]
n = W.shape[1]
'''

# Definition of the Orthogonal matrix
I = np.eye(m, m, dtype = 'd')
Q = np.zeros((m,n), dtype = 'd')

# First column initialization and normalization
qk=W[:, 0]
qk = qk/norm(qk)
Q[:, 0]=qk

orth_loss=[]
condQ=[]
condW=[]

# Application of the sequential algorithm by defining the Projection Matrix P=I-QQ^T
for k in range(1, n):
    P = I - Q@np.transpose(Q)
    qk = P@W[: , k] # Projection
    qk = qk/norm(qk) # Normalization
    Q[: , k] = qk
    II = np.eye(k, k, dtype='d')
    norma = norm(II - np.transpose(Q[:,0:k]) @ Q[:,0:k])
    orth_loss.append(norma)
    condQ.append(np.linalg.cond(Q[:,0:k]))
    condW.append(np.linalg.cond(W[:,0:k]))

# Visualize the Orthogonal Matrix
print(Q)

# Stop evaluating the run-time
wt = time.time() - wt

# Comparison with the numpy.qr operation
wt = time.time()
Q, R=np.linalg.qr(W)
wt = time.time() - wt

# Plot the Condition number of Q
plt.figure()

sns.set_style('darkgrid')
plt.plot(condW, condQ, label='Condition number of Q in [3]', color='green', linestyle='-', marker='o')
plt.title('Condition number of Q vs k(W)')
plt.xlabel('Condition number of W')
plt.ylabel('Condition number of Q')
plt.legend()
plt.savefig('otho[2].png')

plt.figure()

sns.set_style('darkgrid')
plt.plot(condW, orth_loss, label='Condition number of Q in [3]', color='green', linestyle='-', marker='o')
plt.title('Condition number of Q vs k(W)')
plt.xlabel('Condition number of W')
plt.ylabel('Condition number of Q')
plt.legend()
plt.savefig('otho[2].png')

plt.show()
