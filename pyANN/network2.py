"""network2.py
--------------
An improved version of network.py, implementing the SGD learning algrothm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function, regularization, and better 
initialization of network weights.
"""
import numpy as np
import random


class Network():
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            lmbda = 0.0, 
            evaluation_data=None, 
            monitor_evaluation_cost=False, 
            monitor_evaluation_accuracy=False, 
            monitor_training_cost=False, 
            monitor_training_accuracy=False):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        The "training_data" is a list of tuples "(x, y)" representing the traning inputs and the 
            desired outputs.
        'eta' is learning rate.
        'lmbda' is the regularization parameter 'lambda'.
        'evaluation_data' is either the validation or test data.
        We can monitor the cost amnd accuracy on either the validation or training data, by setting the 
            appropriate falgs.
        The method returns a tuple containing four lists:
            the (per-epoch) costs on the evaluation data, 
            the accuracies on the evaluation data, 
            the (per-epoch) costs on the training data, and
            the accuracies on the traning data.
        All values are evaluated at the end of each training epoch.
        So, for example, if we train for epochs, then the first element of the tuple will be a 30-element 
            list containing the cost on the evaluation data at the end of each epoch.
        If the corresponding flag is not set, the lists will be empty.
        """
        if evaluation_data != None:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch {} traning complete".format(j))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {0} / {1}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {0} / {1}".format(accuracy, n))
            print
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation to a 
            single mini batch.
        'mini_batch' is a list of tuples '(x, y)'.
        'eta' is learning rate.
        'lmbda' is the regularization parameter 'lambda'.
        'n' is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y)
        '''
        Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x.
        nabla_b and nabla_w are layer-by-layer lists of numpy arrays, 
        similar to self.biases and self.weights.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop uses a reverse order.
        # Here, l=1 means the last layer of neurons, l=2 is the second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)


class CrossEntropyCost():
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)


class QuadraticCost():
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


def sigmoid_prime(z):
    '''
    Derivative of the sigmoid function.
    '''
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


if __name__ == '__main__':

