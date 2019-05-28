"""weight_initialization.py
---------------------------
This program shows how weight_initialization affects training.
"""
import json
import random
import sys

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network2


def main(filename, n, eta):
    run_network(filename, n, eta)
    make_plots(filename) 


def run_network(filename, n, eta):
    """
    Train the network using both the default and the large starting weights.
    Store the results in the file with name 'filename', where they can later be used by 'make_plots'.
    'n' is the neuron number of the hidden layer.
    'eta' is learning rate.
    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost())
    print("Train the network with the default starting weights.")
    default_vc, default_va, default_tc, default_ta = net.SGD(training_data, 
            30, 10, eta, evaluation_data=validation_data, lmbda=5.0, 
            monitor_evaluation_accuracy=True)
    print("Train the network with the large starting weights.")
    net.large_weight_initializer()
    large_vc, large_va, large_tc, large_ta = net.SGD(training_data, 
            30, 10, eta, evaluation_data=validation_data, lmbda=5.0, 
            monitor_evaluation_accuracy=True)
    with open(filename, "w") as f:
        json.dump({"default_weight_initialization":
                   [default_vc, default_va, default_tc, default_ta], 
                   "large_weight_initialization":
                   [large_vc, large_va, large_tc, large_ta]}, f)

def make_plots(filename):
    """
    Load the results from 'filename', and generate the corresponding plots.
    """
    with open(filename, "r") as f:
        results = json.load(f)
    default_vc, default_va, default_tc, default_ta = results["default_weight_initialization"]
    large_vc, large_va, large_tc, large_ta = results["large_weight_initialization"]
    #  Convert raw classification numbers to percentages, for ploting
    default_va = [x / 100.0 for x in default_va]
    large_va = [x / 100.0 for x in large_va]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, 30, 1), large_va, color='r', label='large weight initialization')
    ax.plot(np.arange(0, 30, 1), default_va, color='b', label='default weight initialization')
    ax.grid(True)
    ax.set_xlim([0, 30])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title("Classification accuracy")
    plt.legend(loc='lower right')
    plt.savefig('./figs/weight_init.png')
    plt.show()


if __name__ == '__main__':
    main("monitor_weights.json", 100, 0.1)

