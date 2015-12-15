# -*- coding:utf-8 -*-
import numpy as np
from scipy.signal import convolve2d
import time

# 克罗内克积
def kronecker(A, B):
    m, n = np.shape(A);
    p, q = np.shape(B);
    K = np.zeros((m * p, n * q));
    
    for i in range(m):
        tmpR = [];
        for j in range(n):
            tmp = A[i, j] * B;
            if j == 0:
                tmpR = tmp;
            else:
                tmpR = np.column_stack((tmpR, tmp));
        if i == 0:
            K = tmpR;
        else:
            K = np.row_stack((K, tmpR));
    
    return K;

# 卷积
def conv(A, B):
    m, n = np.shape(A);
    r, c = np.shape(B);
    
    if r > m or c > n:
        raise Exception("无法卷积");
    
    K = np.zeros((m - r + 1, n - c + 1));
    
    for i in range(m - r + 1):
        for j in range(n - c + 1):
            K[i, j] = np.sum(np.multiply(A[i: i + r, j: j + c], B));
            
    return K;

# 池化
def pooling(A):
    m, n = np.shape(A);
    if (m % 2 != 0) or (n % 2):
        raise Exception("无法池化");
    
    K = np.zeros((m / 2, n / 2));
    
    for i in range(m / 2):
        for j in range(n / 2):
            K[i, j] = np.sum(A[2 * i: 2 * i + 2, 2 * j : 2 * j + 2]) / 4.0;
    
    return K;

A = np.mat([[1 ,1, 1, 1],
     [0, 0, 1, 1],
     [0, 1, 1, 0],
     [0, 1, 1, 0]]);
B = np.mat([[1, 1],
     [0, 1]]);

# kronecker(A, B);
# K = conv(A, B);
# K_2 = convolve2d(A, B, mode='valid');
# print K;
# print K_2
# pooling(A);

c1 = [[1, 2, 3], [1, 2, 3]];
print np.transpose(c1);

