"""
mnist_loader
------------
A library to load the MNIST image data.
"""
import pickle
import gzip
import numpy as np


def load_data():
    """
    Return the MNIST data as a tuple containing the training data, the validation data and the test data.
    The 'training data' is returned as a tuple with two entries.
    The first contains the actual training images.
    This is a numpy ndarray with 50000 entries.
    Each entry is a numpy ndarray with 784 values.
    The second in the 'training data' tuple is a numpy ndarray containing 50000 entries.
    Those entries are just the digital values(0...9) for the corresponding images in the first of the tuple.
    The 'validation data' and 'test data' are similar, except each contains only 10000 images.
    """
    with gzip.open('./mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    Reshape the data to adjust to our neural model.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    Return a 10-demensional unit vector with a 1.0 in the jth position and zeros elsewhere.
    This is used to convert a digit (0...9) into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == '__main__':
    load_data()
