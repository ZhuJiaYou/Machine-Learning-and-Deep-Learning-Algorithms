import sys, os
from struct import unpack

import numpy as np


def read_image(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, rows*cols)  # Add channels' num of pics
    return img


def read_label(path):
    with open(path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        label = np.fromfile(f, dtype=np.uint8)
    return label


def normalize_image(image):
    img = image.astype(np.float32) / 255.0
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
        image_padding[:,zero_num:image.shape[1] + zero_num, zero_num:image.shape[2] + zero_num,:] = image
    elif len(image.shape) == 3:
        image_padding = np.zeros((image.shape[0] + 2 * zero_num, image.shape[1] + 2 * zero_num, 
                                  image.shape[2]))
        image_padding[zero_num:image.shape[0] + zero_num, zero_num:image.shape[1] + zero_num,:] = image
    else:
        print("Error Image Demensions!")
        sys.exit()
    return image_padding


def dataset_loader():
    train_image = read_image("./../datasets/train-images.idx3-ubyte")
    train_label = read_label("./../datasets/train-labels.idx1-ubyte")
    test_image = read_image("./../datasets/t10k-images.idx3-ubyte")
    test_label = read_label("./../datasets/t10k-labels.idx1-ubyte")

    train_image = normalize_image(train_image)
    train_label = one_hot_label(train_label)
#    train_label = train_label.reshape(train_label.shape[0], train_label.shape[1], 1)

    test_image = normalize_image(test_image)
    test_label = one_hot_label(test_label)
#    test_label = test_label.reshape(test_label.shape[0], test_label.shape[1], 1)

#    train_image = padding(train_image, 2)
#    test_image = padding(test_image, 2)

    return train_image, train_label, test_image, test_label


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = dataset_loader()
    print(x_train.shape)
    print(t_train.shape)
