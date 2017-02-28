# coding=utf-8

from __future__ import division
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
import random

np.set_printoptions(precision=4, suppress=True)

data = randn(2, 3)
data.dtype
data.shape

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2
arr2.ndim
print arr2.shape
arr2.dtype

np.zeros(10)
np.zeros((3, 6))
np.empty((2, 3, 2))
np.arange(15)

arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype

arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr1.dtype
arr2.dtype

arr = np.array([[1, 2, 3], [4, 5, 6]])
arr
arr * arr
arr - arr

arr.dtype
arr.dtype

arr = np.arange(10)
arr[6:9]

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[1, :2]

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = randn(7, 4)
data
data[names == 'Bob', 2:]

arr = np.empty((4, 4))
for i in range(4):
    arr[i] = i
arr
arr.T

x = randn(7)
y = randn(7)
x
y
np.add(x, y)

# points = np.arange(-5, 5, 0.01)
# 产生两个二维矩阵
# xs, ys = np.meshgrid(points, points)
# xs
# ys
# z = np.sqrt(xs ** 2 + ys ** 2)
# plt.imshow(z, cmap=plt.cm.gray)
# plt.colorbar()
# plt.show()

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
# odd
# result = [(x if c else y)
#           for x, y, c in zip(xarr, yarr, cond)]
# result

result = np.where(cond, xarr, yarr)

arr = randn(4, 4)
arr = np.where(arr > 0, 2, arr)

arr = randn(5, 4)
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
(arr > 0).sum()
arr.cumsum(0)
arr.cumprod(1)

arr = randn(5, 3)
arr
arr.sort()
arr


large_arr = range(100)
large_arr
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]  # 5% quantile

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)

sorted(set(names))
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

# arr = np.arange(10)
# np.save('some_array.npy', arr)
# print np.load('some_array.npy')

arr = np.loadtxt('input_data.csv', delimiter=',')

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
# print x.dot(y)  # equivalently np.dot(x, y)
# print np.dot(x, np.ones(3))

X = randn(5, 5)
mat = X.T.dot(X)
# print mat.dot(inv(mat))  # 求逆矩阵


np.random.seed(12345)
nsteps = 100
nwalks = 100
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
# print walk.min()
# print walk.max()
# argmax函数 返回walk数组第一次出现最大值的index，需要扫描整个数组。
# print (np.abs(walk) >= 10).argmax()

# arr = np.randn(5, 5)
# print (arr > 0).any(1)

steps = np.random.normal(loc=0, scale=0.25,
                         size=(nwalks, nsteps))
# print steps
