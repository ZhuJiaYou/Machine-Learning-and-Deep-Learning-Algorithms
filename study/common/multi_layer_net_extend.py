import numpy as np
from collections import OrderedDict

from common.gradient import numerical_gradient
from common.layers import *


class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', 
                 weight_init_std='relu', weight_decay_lambda=0, 
                 use_dropout=False, dropout_ratio=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm

        self.__init_weight(weight_init_std)

        activation_layer = {'Sigmoid':Sigmoid, 'relu': Relu}

        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)] = Affine(self.params['w'+str(idx)], self.params['b'+str(idx)])
            if self.use_batchnorm:
                self.params['gamma'+str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta'+str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm'+str(idx)] = BatchNormalization(self.params['gamma'+str(idx)], 
                                                                       self.params['beta'+str(idx)])
            self.layers['Activation_func'+str(idx)] = activation_layer[activation]()
            if self.use_dropout:
                self.layers['Dropout'+str(idx)] = Dropout(dropout_ratio)
        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(self.params['w'+str(idx)], self.params['b'+str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params['w'+str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            w = self.params['w'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(w ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t, train_flg=False)
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['w'+str(idx)] = numerical_gradient(loss_w, self.params['w'+str(idx)])
            grads['b'+str(idx)] = numerical_gradient(loss_w, self.params['b'+str(idx)])
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma'+str(idx)] = numerical_gradient(loss_w, self.params['gamma'+str(idx)])
                grads['beta'+str(idx)] = numerical_gradient(loss_w, self.params['beta'+str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['w'+str(idx)] = self.layers['Affine'+str(idx)].dw + 
                                  self.weight_decay_lambda * self.layers['Affine'+str(idx)].w
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db
            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads['gamma'+str(idx)] = self.layers['BatchNorm'+str(idx)].dgamma
                grads['beta'+str(idx)] = self.layers['BatchNorm'+str(idx)].dbeta
        return grads



if __name__ == '__main__':
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    net = SimpleNet()

    f = lambda w: net.loss(x, t)
    dw = numerical_gradient(f, net.w)
    print(dw)
