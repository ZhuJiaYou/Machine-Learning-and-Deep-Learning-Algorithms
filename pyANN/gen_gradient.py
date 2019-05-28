"""gen_gradient.py
------------------
Use network2 to figure out the average starting values of the gradient error terms
\delta^l_j = \partial c / \partial z^l_j = \partial c / \partial b^l_j
"""
import json
import math
import random
import shutil  # d高级的 文件、文件夹、压缩包 处理模块
import sys

import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network2


def main():
    full_training_data = mnist_loader.load_data_wrapper()
    training_data = full_training_data[:1000]
    epoches = 500

    print("\nTwo hidden layers:")
    net = network2.Network([784, 30, 30, 10])
    initial_norms(training_data, net)
    abbreviated_gradient = [ag[:6] for ag in get_average_gradient(net, td)[:-1]]
    print("Saving the averaged gradient for the top six nerons in each layer.")
    with open("initial_gradient.json", "w"):
        json.dump(abbreviated_gradient, f)
    shutil.copy("initial_gradient.json", "./js/initial_gradient.json")
    plot_training(epoches, "norms_during_training_2_layers.json", 2)


def initial_norms(training_data, net):
    average_gradient = 
