import numpy
import numpy as np
import torch
# f = np.loadtxt('./000001.txt')
# print(f)
# print(len(f))
# print(len(f.shape))
#
# x = torch.zeros(10, 3, 13, 13, 4)
# print(x)
#
# arr = np.eye(4, 4)
# print(arr)

ndim = 4
dt = 1
_motion_mat = np.eye(2 * ndim, 2 * ndim)
for i in range(ndim):
    _motion_mat[i, ndim + i] = dt
    _update_mat = np.eye(ndim, 2 * ndim)

print(_motion_mat)
print(_update_mat)