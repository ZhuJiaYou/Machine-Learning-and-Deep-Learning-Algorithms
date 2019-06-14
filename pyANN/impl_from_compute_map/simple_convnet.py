import pickle
import numpy as np
from collections import OrderedDict
from common.layers import Convolution, Affine, SoftmaxWithLoss, Relu, Pooling
from common.gradient import numerical_gradient


class SimpleConvnet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2)**2)

        # weights init
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(filter_num, input_dim[0],
                                                              filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['w2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['w3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # gen layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['w1'], self.params['b1'], conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['w2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)  # NOQA
        grads = {}
        for idx in (1, 2, 3):
            grads['w'+str(idx)] = numerical_gradient(loss_w, self.params['w'+str(idx)])
            grads['b'+str(idx)] = numerical_gradient(loss_w, self.params['b'+str(idx)])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # settings
        grads = {}
        grads['w1'], grads['b1'] = self.layers['Conv1'].dw, self.layers['Conv1'].db
        grads['w2'], grads['b2'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
        grads['w3'], grads['b3'] = self.layers['Affine2'].dw, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].w = self.params['w'+str(i+1)]
            self.layers[key].b = self.params['b'+str(i+1)]
