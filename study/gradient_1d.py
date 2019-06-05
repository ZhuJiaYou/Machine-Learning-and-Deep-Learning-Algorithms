import sys, os
import numpy as np
import matplotlib.pylab as plt
from data_loader import dataset_loader


def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size


def cross_entropy_error2(y, t):
    batch_size = y.shape[0]
    print(batch_size)
    print([np.arange(batch_size), t])
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def func_1(x):
    return 0.01*x**2 + 0.1*x


def func_2(x):
    return x[0]**2 + x[1]**2


def tangent_line(f, x):
    slope = numerical_diff(f, x)
    bias = f(x) - slope*x
    return lambda t: slope*t + bias


if __name__ == '__main__':
#    x_train, t_train, x_test, t_test = dataset_loader()
#    t2 = np.array([0, 1, 2])

#    t = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
#                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

#    y = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
#                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
#    print(t.reshape(1, t.size))
#    print(cross_entropy_error2(y, t2))
#    print(cross_entropy_error(y, t))

    x = np.arange(0.0, 20.0, 0.1)
    y = func_1(x)
    y2 = tangent_line(func_1, 10)(x)
#    plt.xlabel("x")
#    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.plot(x, y)
    plt.show()



