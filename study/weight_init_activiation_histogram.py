import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


input_data = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activiations = {}
x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activiations[i-1]

#    w = np.random.randn(node_num, node_num) * 1
#    w = np.random.randn(node_num, node_num) * 0.01
#    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
    z = np.dot(x, w)
#    a = sigmoid(z)
    a = ReLU(z)
#    a = tanh(z)
    activiations[i] = a

for i, a in activiations.items():
    plt.subplot(1, len(activiations), i+1)
    plt.title("{}-layer".format(i+1))
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
