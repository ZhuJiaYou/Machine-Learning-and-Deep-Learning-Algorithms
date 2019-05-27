"""overfitting.py
--------------
Plot graphs to illustrate the problem of overfitting.
"""
import json
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network2


def main(filename, num_epoches, 
         training_cost_xmin=200, 
         test_accuracy_xmin=200, 
         test_cost_xmin=0, 
         training_accuracy_xmin=0, 
         training_set_size=1000, 
         lmbda=0.0):
    """
    'filename' is the name of the file where the results will be stored.
    'num_epoches' is the number of epoches to train for.
    'training_set_size' is the number of images to train on.
    'lmbda' is the regularization parameter.
    The other parameters set the epoches at which to start plotting on the x axis.
    """
    run_network(filename, num_epoches, training_set_size, lmbda)
    make_plots(filename, num_epoches, training_cost_xmin, test_accuracy_xmin, test_cost_xmin, 
               training_accuracy_xmin, training_set_size)


def run_network(filename, num_epoches, training_set_size=1000, lmbda=0.0):
    """
    Train the network for 'num_epoches' on 'training_set_size' images, and store the results in 'filename'.
    Those results can later be used by 'make_plots'.
    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())
    net.large_weight_initializer()
    test_cost, test_accuracy, training_cost, training_accuracy = net.SGD(training_data[:training_set_size], 
            num_epoches, 10, 0.5, evaluation_data=test_data, lmbda=lmbda, 
            monitor_evaluation_accuracy=True, 
            monitor_evaluation_cost=True, 
            monitor_training_cost=True, 
            monitor_training_accuracy=True)
    with open(filename, "w") as f:
        json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)

def make_plots(filename, num_epoches, 
               training_cost_xmin=200, 
               test_accuracy_xmin=200, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0, 
               training_set_size=1000):
    """
    Load the results from 'filename', and generate the corresponding plots.
    """
    with open(filename, "r") as f:
        test_cost, test_accuracy, training_cost, training_accuracy = json.load(f)
    plot_training_cost(training_cost, num_epoches, training_cost_xmin)
    plot_test_accuracy(test_accuracy, num_epoches, test_accuracy)
    plot_training_accuracy(training_accuracy, num_epoches, training_accuracy_xmin, training_set_size)
    plot_overlay(test_accuracy, training_accuracy, num_epoches, 
                 min(test_accuracy_xmin, training_accuracy_xmin), training_set_size)


def plot_training_cost(training_cost, num_epoches, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epoches), training_cost[training_cost_xmin:num_epoches], 
                      color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epoches])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()


def plot_test_cost(test_cost, num_epoches, test_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epoches), test_cost[test_cost_xmin:num_epoches], 
                      color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epoches])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()


def plot_test_accuracy(test_accuracy, num_epoches, test_accuracy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epoches), 
            [accuracy / 100.0 for accuracy in test_accuracy[test_accuracy_xmin:num_epoches]],
            color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epoches])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy(%) on the test data')
    plt.show()


def plot_training_accuracy(training_accuracy, num_epoches, training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epoches), 
            [accuracy * 100.0 / training_set_size 
                for accuracy in training_accuracy[training_accuracy_xmin:num_epoches]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epoches])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy(%) on the training data')
    plt.show()


def plot_overlay(test_accuracy, training_accuracy, num_epoches, xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epoches), 
            [accuracy / 100.0 for accuracy in test_accuracy],
            color='#2A6EA6', 
            label='Accuracy on the test data')
    ax.plot(np.arange(xmin, num_epoches), 
            [accuracy * 100.0 / training_set_size for accuracy in training_accuracy],
            color='#FFA933', 
            label='Accuracy on the training data'))
    ax.set_xlim([test_accuracy_xmin, num_epoches])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy(%) on the test data')
    plt.show()


if __name__ == '__main__':

