import numpy as np


def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :]
    t = t[permutation]
    return x, t


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    patameters
    ----------
    input_data: 4 dimensional(data num, channels, hight, width) array
    filter_h: hight of the filter(convolution kernel)
    filter_w: width of the filter

    returns
    -------
    col: 2 dimensional array
    """


