import numpy as np
import sys
import math
from numpy.linalg import inv

# perform linear regression to predict the sujectively reported
# hunger based on the electrode data. Write code to calculate the
# psuedo-inverse yourself (don't use library call).

# psuedo inverse assuming linearly independent columns of A
def psuedo_inverse(A):
    return inv(A.T.dot(A)).dot(A.T)

# MSE returns the mean-squared error between sets A and B
def MSE(A, B):
    return ((A-B)**2).mean()

data = np.array([[1,     8,   3], \
                 [-46, -98, 108], \
                 [5,    12,  -9], \
                 [63,  345, -27], \
                 [23,   78,  45], \
                 [-12,  56,  -8], \
                 [1,    34,  78], \
                 [56,  123,  -5]])

hunger = np.array([65.66, -1763.1, 195.2, 3625, 716.9, 339, -25.5, 1677.1])

x = np.dot(psuedo_inverse(data), hunger)
print("Coeffficients:")
print(x)

predict = np.dot(data,x)
mse = MSE(hunger, predict)

print("actual hunger:")
print(hunger)
print("linear regression prediction:")
print(predict)
print("Mean Squared Error:")
print(mse)