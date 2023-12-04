# Disable numpy threading to get useful timings
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Import Libraries
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy import sin
from numpy import cos

# Define discretizing function
def f(mu, x):
    result = np.sin(10*(mu+x))/(np.cos(100*(mu-x))+1.1)
    return result

# Initialization of MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Declaration of the matrices
W = None
Q = None
QT = None
q_local= None

if rank == 0:
    # Definition of the Matrix
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
    W = np.arange(1, m*n + 1, 1, dtype='d')
    W = np.reshape(W, (m, n))
    W = W + np.eye(m, n)
    '''
    # Case 3:
    '''data = loadmat('full_matrix.mat')
    W = data['good_cond']
    m = W.shape[0]
    n = W.shape[1]
    '''
    local_size = int(m/size)
    #W = W.astype(np.float64) for the Case 3 import
    Q = np.zeros((m, n), dtype='d')
    condQ=[]
    condW=[]
    ortho=[]
else:
    m = None
    n = None
    local_size = None

# Communicate dimensions
m = comm.bcast(m, root=0)
n = comm.bcast(n, root=0)
local_size = comm.bcast(local_size, root=0)

# Definition and normalization of the first column
if rank==0:
    q_temp = W[:, 0]
    q_temp = q_temp / norm(q_temp)
    Q[:,0]=q_temp

# We are going to scatter and broadcast the values to the processors
W_local = np.empty((local_size, n), dtype=np.float64)
# We scatter the matrix W by rows
comm.Scatterv(W, W_local, root=0)

Q_local = np.empty((local_size, n), dtype='d')


for k in range(1, n):
    q_aux = np.zeros(n, dtype='d')
    comm.Scatterv(Q, Q_local, root=0)
    q_local = np.transpose(Q_local)@W_local[:, k]
    comm.Allreduce(q_local, q_aux, op=MPI.SUM)
    q_local = W_local[:, k] - Q_local@q_aux
    q_temp = np.zeros(m, dtype='d')
    comm.Gatherv(q_local,q_temp,root=0)
    # Every time we scatter the values, we resume them in rank 0
    if rank == 0:
        q_temp = q_temp / norm(q_temp)
        Q[:, k] = q_temp
        I=np.eye(k,k)
        ortho.append(norm(I-np.transpose(Q[:,0:k])@Q[:,0:k]))
        condW.append(np.linalg.cond(W[:,0:k]))
        condQ.append(np.linalg.cond(Q[:,0:k]))


# Plot the Condition Number of Q
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
plt.plot(condW, ortho, label='Condition number of Q in [3]', color='green', linestyle='-', marker='o')
plt.title('Condition number of Q vs k(W)')
plt.xlabel('Condition number of W')
plt.ylabel('Condition number of Q')
plt.legend()
plt.savefig('otho[2].png')

plt.show()