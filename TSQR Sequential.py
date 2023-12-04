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

# Discretizing function
def f(mu, x):
    result = np.sin(10*(mu+x))/(np.cos(100*(mu-x))+1.1)
    return result

# Define the number of blocks to divide the matrix
block = 4

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

local_size = int(m/block)

# QR factorization for the first row block
Matrix0 = W[0:local_size, :]
Q_fin, R = np.linalg.qr(Matrix0)

# For-loop to iterate over all blocks
for i in range(local_size, m, local_size):
    # Stack the matrices to obtain the new matrix
    Matrix0 = np.vstack((R, W[i:(i+local_size), :]))
    Q, R = np.linalg.qr(Matrix0)
    # Fix the dimensions of the Q_i to multiply
    Q_fin = np.block([[Q_fin, np.zeros((i, local_size))],
                      [np.zeros((local_size, n)), np.eye(local_size, local_size)]])
    Q_fin=Q_fin@Q

print(np.linalg.cond(Q_fin))
I=np.eye(n,n)
norma=norm(I-np.transpose(Q_fin)@Q_fin)
print(norma)
print(np.linalg.cond(W))