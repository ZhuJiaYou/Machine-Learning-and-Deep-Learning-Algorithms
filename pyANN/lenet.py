"""lenet.py
Implementation of LeNet
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import matplotlib
# matplotlib.use('Agg')


def read_image(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols, 1)  # Add channels' num of pics
    return img


def read_label(path):
    with open(path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        label = np.fromfile(f, dtype=np.uint8)
    return label


def normalize_image(image):
    img = img.astype(np.float32) / 255.0
    return img


def one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def padding(image, zero_num):
    if len(image.shape) == 4:
        image_padding = np.zeros((image.shape[0], image.shape[1] + 2 * zero_num, 
                                  image.shape[2] + 2 * zero_num, image.shape[3]))
        image_padding[:,zero_num:image.shape[1] + zero_num, zero_num:image.shape[2] + zero_num] = image
    elif len(image.shape) == 3:
        image_padding = np.zeros((image.shape[0] + 2 * zero_num, image.shape[1] + 2 * zero_num, 
                                  image.shape[2]))
        image_padding[zero_num:image.shape[0] + zero_num, zero_num:image.shape[1] + zero_num] = image
    else:
        print("Error Image Demensions!")
        sys.exit()
    return image_padding


def dataset_loader():
    print("aaabbbccc")


def conv(img, conv_filter):
    if len(img.shape) != 3 or len(conv_filter.shape) != 4:
        print("Error Dimension of the Conv Op!")
        sys.exit()
    if img.shape[-1] != conv_filter.shape[-1]:
        print("Channel of the img should be the same as conv filter!")
        sys.exit()
    img_h, img_w, img_ch = img.shape
    filter_num, filter_h, filter_w, img_ch = conv_filter.shape


def soft_max(z):
    tmp = np.max(z)
    z -= tmp
    z = np.exp(z)
    tmp = np.sum(z)
    z /= tmp

    return z


