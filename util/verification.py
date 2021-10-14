import numpy as np
import torch

# epsilon = 1e-4
# x = torch.randn(2, 3)
# print('x:', x)
# x_2 = torch.mm(x.t(), x)
# print(x.shape[1])
# x_2 += torch.eye(x.shape[1]) * epsilon
# L = torch.cholesky(x_2)
# # 最后乘以的根号数字是什么
# print(x.shape[0])
# y = torch.tensor(x.shape[0])
# y=y.float()
# print('y:', y)
# ortho_weights = (torch.inverse(L)).t() * torch.sqrt(y)
# print(type(ortho_weights))
# print(ortho_weights)
# print(torch.inverse(L))
# print((torch.inverse(L)).t())

#
X = np.array([[1, 1, 1 ,1],
              [2,2,2,2],
              [3,3,3,3]])
print(X.shape)
Y = None
W = None
if Y is None:
    Y = X
# distance = squaredDistance(X, Y)
# K.ndim(X)返回X的轴数
sum_dimensions = list(range(2, np.ndim(X) + 1))
print(sum_dimensions)
# 维数扩充 axis=1表示扩充为nX1Xm维 axis=0表示扩充为1XnXm维
X = np.expand_dims(X, axis=1)
print('extand_X:', X)
print('shape_of_eextandX:', X.shape)
if W is not None:
    # if W provided, we normalize X and Y by W
    # axis=1表示按行求和，axis=0表示按列求和
    D_diag = np.expand_dims(np.sqrt(np.sum(W, axis=1)), axis=1)
    X /= D_diag
    Y /= D_diag
print(X-Y)
print((X-Y).shape)
squared_difference = np.square(X - Y)
distance = np.sum(squared_difference, axis=sum_dimensions[0])
print(distance)
print(distance.shape)