import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient, func_2


def func_3(x):
    return (1/20)*(x[0]**2) + x[1]**2


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x, np.array(x_history)


if __name__ == '__main__':
    init_x = np.array([-7.0, 2.0])
    step_num = 100
    lr = 0.8
    x, x_history = gradient_descent(func_3, init_x, lr=lr, step_num=step_num)

    plt.plot([-10, 10], [0, 0], '--b')
    plt.plot([0, 0], [-10, 10], '--b')
    plt.plot(x_history[:, 0], x_history[:, 1], "o")
#    plt.xlim(-3.5, 3.5)
#    plt.ylim(-4.5, 4.5)
    plt.xlabel("x0")
    plt.xlabel("x1")
    plt.show()
