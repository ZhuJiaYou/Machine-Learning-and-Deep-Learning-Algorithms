"""network3.py
--------------
A Theano-based program for training and running simple neural networks.
Supports several layer types(fully connected, convolutional, max pooling, softmax), and activation functions
    (sigmoid, tanh, and rectified linear units, with more easily added).
When run on a CPU, this program is much faster than network.py and network2.py, and can also be run on a GPU.



"""
import numpy as np


class ReluActiviator(object):
    def forward(self, weighted_input):
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0


class SigmoidActiviator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return 
