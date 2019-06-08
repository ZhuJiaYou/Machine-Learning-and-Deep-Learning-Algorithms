import numpy as np
import matplotlib.pylab as plt
from common.multi_layer_net_extend import MultiLayerNetExtend 
from data_loader import dataset_loader


x_train, t_train, x_test, t_test = dataset_loader()
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100], output_size=10, 
                               use_batchnorm=True)
x_batch = x_train[:1]
t_batch= t_train[:1]

grad_backprop = network.gradient(x_batch, t_batch)
grad_numerical = network.numerical_gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + " == {}".format(diff))
