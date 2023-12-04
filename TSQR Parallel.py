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

# Discretizing function
def f(mu, x):
    result = np.sin(10*(mu+x))/(np.cos(100*(mu-x))+1.1)
    return result

# Define the reduction operation
def QR_tilde_complete_reduction(W_0, W_1, datatype):
    # Split the matrices since they arrive stack in a block
    R_0, Q_0 = np.split(W_0, [W_0.shape[1]], axis=0)
    R_1, Q_1 = np.split(W_1, [W_1.shape[1]], axis=0)
    # Stack the R matrices
    R_stack = np.vstack((R_0, R_1))
    m_dim = Q_0.shape[0]
    n_dim = Q_0.shape[1]
    # Compute the QR factorization to the new matrix
    Q, R = np.linalg.qr(R_stack)
    Q_fin = np.zeros((2 * m_dim, 2 * n_dim))
    Q_fin[0:m_dim, 0:n_dim] = Q_0
    Q_fin[m_dim:, n_dim:] = Q_1
    Q_fin = Q_fin @ Q
    # Return the stack matrix
    return_matrix = np.vstack((R, Q_fin))
    return return_matrix

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

W = None

if rank == 0:
    # Definition of the matrix
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
    '''
        data = loadmat('full_matrix.mat')
        W = data['good_cond']
        m = W.shape[0]
        n = W.shape[1]
        '''
else:
    m = None
    n = None

m = comm.bcast(m,root=0)
n = comm.bcast(n,root=0)
local_size = int(m / size)

# Scatter the block rows
W_local = np.zeros((local_size, n), dtype=np.float64)
comm.Scatter(W, W_local, root=0)

# Computation the first QR factorization to all the processors
Q_loc, R_loc = np.linalg.qr(W_local)
M_loc = np.vstack((R_loc, Q_loc))

# Creation of the reduction operation
QR_op = MPI.Op.Create(QR_tilde_complete_reduction, commute=False)

# Reduction to the processors zero
R_and_Q = comm.reduce(M_loc, op=QR_op, root=0)

if rank == 0:
    R_final, Q_final = np.split(R_and_Q, [n], axis=0)
    I = np.eye(n, n)
    norma=norm(I - np.transpose(Q) @ Q)
    condW=np.linalg.cond(W)
    condQ=np.linalg.cond(Q)

