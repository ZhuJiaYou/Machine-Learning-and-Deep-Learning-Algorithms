import numpy as np
from data_loader import dataset_loader
from two_layer_net_with_layers import TwoLayerNet


x_train, t_train, x_test, t_test = dataset_loader()
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
x_batch = x_train[:3]
t_batch = t_train[:3]
gradient_numerical = network.numerical_gradient(x_batch, t_batch)
gradient_backprop = network.gradient(x_batch, t_batch)

for key in gradient_numerical.keys():
    diff = np.average(np.abs(gradient_backprop[key] - gradient_numerical[key]))
    print(key + ":" + str(diff))

