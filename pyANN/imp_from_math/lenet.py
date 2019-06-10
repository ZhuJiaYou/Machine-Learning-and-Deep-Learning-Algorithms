"""lenet.py
Implementation of LeNet
"""
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack


class ConvNet:
    def __init__(self):
        """
        Two Conv Layers, Two Pooling and Three Fully Connected.
        """
        self.filters = [np.random.randn(6, 5, 5, 1)]
        self.filters_biases = [np.random.randn(6, 1)]
        self.filters.append(np.random.randn(16, 5, 5, 6))
        self.filters_biases.append(np.random.randn(16, 1))

        self.weights = [np.random.randn(120, 400)]
        self.weights.append(np.random.randn(84, 120))
        self.weights.append(np.random.randn(10, 84))
        self.biases = [np.random.randn(120, 1)]
        self.biases.append(np.random.randn(84, 1))
        self.biases.append(np.random.randn(10, 1))

    def feed_forward(self, x):
        conv1 = add_bias(conv(x, self.filters[0]), self.filters_biases[0])
        relu1 = relu(conv1)
        pool1, pool1_max_locate = pool(relu1)

        conv2 = add_bias(conv(pool1, self.filters[1]), self.filters_biases[1])
        relu2 = relu(conv2)
        pool2, pool2_max_locate = pool(relu2)

        straight_input = pool2.reshape(pool2.shape[0] * pool2.shape[1] * pool2.shape[2], 1)

        full_connect1_z = np.dot(self.weights[0], straight_input) + self.biases[0]
        full_connect1_a = relu(full_connect1_z)

        full_connect2_z = np.dot(self.weights[1], full_connect1_a) + self.biases[1]
        full_connect2_a = relu(full_connect2_z)

        full_connect3_z = np.dot(self.weights[2], full_connect2_a) + self.biases[2]
        full_connect3_a = soft_max(full_connect3_z)

        return full_connect3_a

    def SGD(self, train_image, train_label, test_image, test_label, epoches, mini_batch_size, eta):
        batch_num = 0
        fx = []
        fy_loss = []
        fy_accuracy = []
        for j in range(epoches):
            mini_batch_image = [train_image[k:k+mini_batch_size] 
                                for k in range(0, len(train_image), mini_batch_size)]
            mini_batch_label = [train_label[k:k+mini_batch_size] 
                                for k in range(0, len(train_label), mini_batch_size)]
            for mini_batch_image, mini_batch_label in zip(mini_batch_image, mini_batch_label):
                batch_num += 1
                if batch_num * mini_batch_size > len(train_image):
                    batch_num = 1
                self.update_mini_batch(mini_batch_image, mini_batch_label, eta, mini_batch_size)
                print("\rEpoch{0}:{1}/{2}".format(j+1, batch_num*mini_batch_size, len(train_image)), end="")
            accurate_num, loss = self.evaluate(test_image, test_label)
            plt.figure(1)
            fx.append(j)
            fy_accuracy.append((0.0 + accurate_num) / len(test_image))
            fy_loss.append(loss)
            print(" After epoch{0}: accuracy is {1}/{2}, loss is {3}".format(j+1, accurate_num, 
                                                                            len(test_image), loss))
        my_x_ticks = np.arange(1, epoches+1, 1)
        plt.figure(1)
        plt.xlabel("Epoches")
        plt.ylabel("loss")
        plt.xticks(my_x_ticks)
        plt.plot(fx, fy_loss, "bo-")

        plt.figure(2)
        plt.xlabel("Epoches")
        plt.ylabel("accuracy")
        plt.xticks(my_x_ticks)
        plt.plot(fx, fy_accuracy, "r+-")
        plt.show()

    def update_mini_batch(self, mini_batch_image, mini_batch_label, eta, mini_batch_size):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_f = [np.zeros(f.shape) for f in self.filters]
        nabla_fb = [np.zeros(fb.shape) for fb in self.filters_biases]

        for x, y in zip(mini_batch_image, mini_batch_label):
            delta_nabla_w, delta_nabla_b, delta_nabla_f, delta_nabla_fb = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_f = [nf + dnf for nf, dnf in zip(nabla_f, delta_nabla_f)]
            nabla_fb = [nfb + dnfb for nfb, dnfb in zip(nabla_fb, delta_nabla_fb)]

        self.weights = [w-(eta/mini_batch_size)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases= [b-(eta/mini_batch_size)*nb for b, nb in zip(self.biases, nabla_b)]
        self.filters = [f-(eta/mini_batch_size)*nf for f, nf in zip(self.filters, nabla_f)]
        self.filters_biases = [fb-(eta/mini_batch_size)*nfb 
                               for fb, nfb in zip(self.filters_biases, nabla_fb)]

    def backprop(self, x, y):
        conv1 = add_bias(conv(x, self.filters[0]), self.filters_biases[0])
        relu1 = relu(conv1)
        pool1, pool1_max_locate = pool(relu1)

        conv2 = add_bias(conv(pool1, self.filters[1]), self.filters_biases[1])
        relu2 = relu(conv2)
        pool2, pool2_max_locate = pool(relu2)

        straight_input = pool2.reshape(pool2.shape[0] * pool2.shape[1] * pool2.shape[2], 1)

        full_connect1_z = np.dot(self.weights[0], straight_input) + self.biases[0]
        full_connect1_a = relu(full_connect1_z)

        full_connect2_z = np.dot(self.weights[1], full_connect1_a) + self.biases[1]
        full_connect2_a = relu(full_connect2_z)

        full_connect3_z = np.dot(self.weights[2], full_connect2_a) + self.biases[2]
        full_connect3_a = soft_max(full_connect3_z)

        delta_fc3 = full_connect3_a - y
        delta_fc2 = np.dot(self.weights[2].transpose(), delta_fc3) * relu_prime(full_connect2_z)
        delta_fc1 = np.dot(self.weights[1].transpose(), delta_fc2) * relu_prime(full_connect1_z)
        delta_straight_input = np.dot(self.weights[0].transpose(), delta_fc1)
        delta_pool2 = delta_straight_input.reshape(pool2.shape)
        delta_conv2 = pool_delta_error_bp(delta_pool2, pool2_max_locate) * relu_prime(conv2)
        delta_pool1 = conv(padding(delta_conv2, self.filters[1].shape[1]-1), 
                           rot180(self.filters[1]).swapaxes(0, 3))
        delta_conv1 = pool_delta_error_bp(delta_pool1, pool1_max_locate) * relu_prime(conv1)

        nabla_w2 = np.dot(delta_fc3, full_connect2_a.transpose())
        nabla_b2 = delta_fc3
        nabla_w1 = np.dot(delta_fc2, full_connect1_a.transpose())
        nabla_b1 = delta_fc2
        nabla_w0 = np.dot(delta_fc1, straight_input.transpose())
        nabla_b0 = delta_fc1

        nabla_filters1 = conv_cal_w(delta_conv2, pool1)
        nabla_filters_biases1 = conv_cal_b(delta_conv2)
        nabla_filters0 = conv_cal_w(delta_conv1, x)
        nabla_filters_biases0 = conv_cal_b(delta_conv1)

        nabla_w = [nabla_w0, nabla_w1, nabla_w2]
        nabla_b = [nabla_b0, nabla_b1, nabla_b2]
        nabla_f = [nabla_filters0, nabla_filters1]
        nabla_fb = [nabla_filters_biases0, nabla_filters_biases1]

        return nabla_w, nabla_b, nabla_f, nabla_fb

    def evaluate(self, images, labels):
        result = 0  # record of accuracy
        J = 0  # record of loss
        eta = 1e-7  # preventing overflow of log computation
        for img, lab in zip(images, labels):
            predict_label = self.feed_forward(img)
            if np.argmax(predict_label) == np.argmax(lab):
                result += 1
            J = J + sum(-lab*(np.log(predict_label+eta)) - (1-lab)*np.log(1-predict_label+eta))
        return result, J


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
    train_label = train_label.reshape(train_label.shape[0], train_label.shape[1], 1)

    test_image = normalize_image(test_image)
    test_label = one_hot_label(test_label)
    test_label = test_label.reshape(test_label.shape[0], test_label.shape[1], 1)

    train_image = padding(train_image, 2)
    test_image = padding(test_image, 2)

    return train_image, train_label, test_image, test_label
    


def conv(img, conv_filter):
    if len(img.shape) != 3 or len(conv_filter.shape) != 4:
        print("Error Dimension of the Conv Op!")
        sys.exit()
    if img.shape[-1] != conv_filter.shape[-1]:
        print("Channel of the img should be the same as conv filter!")
        sys.exit()

    img_h, img_w, img_ch = img.shape
    filter_num, filter_h, filter_w, img_ch = conv_filter.shape
    feature_h = img_h - filter_h + 1
    feature_w = img_w - filter_w + 1
    
    img_out = np.zeros((feature_h, feature_w, filter_num))
    img_matrix = np.zeros((feature_h*feature_w, filter_h*filter_w*img_ch))
    filter_matrix = np.zeros((filter_h*filter_w*img_ch, filter_num))

    for j in range(img_ch):
        img_2d = np.copy(img[:,:,j])
        shape = (feature_h, feature_w, filter_h, filter_w)
        strides = (img_w, 1, img_w, 1)
        strides = img_2d.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols = x_cols.reshape(feature_h*feature_w, filter_h*filter_w)
        img_matrix[:,j*filter_h*filter_w:(j+1)*filter_h*filter_w] = x_cols

    for i in range(filter_num):
        filter_matrix[:,1] = conv_filter[i,:].transpose(2,0,1).reshape(filter_w*filter_h*img_ch)

    feature_matrix = np.dot(img_matrix, filter_matrix)

    for i in range(filter_num):
        img_out[:,:,i] = feature_matrix[:,i].reshape(feature_h, feature_w)

    return img_out


def conv_cal_w(out_img_delta, in_img):
    img_h, img_w, img_ch = in_img.shape
    feature_h, feature_w, filter_num = out_img_delta.shape
    filter_h = img_h - feature_h + 1
    filter_w = img_w - feature_w + 1

    in_img_matrix = np.zeros([filter_h*filter_w*img_ch, feature_h*feature_w])
    out_img_delta_matrix = np.zeros([feature_h*feature_w, filter_num])
    
    for j in range(img_ch):
        img_2d = np.copy(in_img[:,:,j])
        shape = (filter_h, filter_w, feature_h, feature_w)
        strides = (img_w, 1, img_w, 1)
        strides = img_2d.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols = x_cols.reshape(filter_h*filter_w, feature_h*feature_w)
        in_img_matrix[j*filter_h*filter_w:(j+1)*filter_h*filter_w,:] = x_cols

    for i in range(filter_num):
        out_img_delta_matrix[:,i] = out_img_delta[:,:,i].reshape(feature_h*feature_w)

    filter_matrix = np.dot(in_img_matrix, out_img_delta_matrix)
    nabla_conv = np.zeros([filter_num, filter_h, filter_w, img_ch])

    for i in range(filter_num):
        nabla_conv[i,:] = filter_matrix[:,i].reshape(img_ch, filter_h, filter_w).transpose(1,2,0)

    return nabla_conv


def conv_cal_b(out_img_delta):
    nabla_b = np.zeros((out_img_delta.shape[-1],1))
    for i in range(out_img_delta.shape[-1]):
        nabla_b[i] = np.sum(out_img_delta[:,:,i])
    return nabla_b


def relu(feature):
    return feature * (feature > 0)


def relu_prime(feature):
    return 1 * (feature > 0)


def pool(feature, size=2, stride=2):
    feature_h, feature_w, feature_ch = feature.shape
    pool_h = np.uint16((feature_h - size) / stride + 1)
    pool_w = np.uint16((feature_w - size) / stride + 1)
    feature_reshaped = feature.reshape(pool_h, feature_h//pool_h, pool_w, feature_w//pool_w, feature_ch)
    out = feature_reshaped.max(axis=1).max(axis=2)
    out_location_c = feature_reshaped.max(axis=1).argmax(axis=2)
    out_location_r = feature_reshaped.max(axis=3).argmax(axis=1)
    out_location = out_location_r * size + out_location_c

    return out, out_location


def pool_delta_error_bp(pool_out_delta, pool_out_max_location, size=2, stride=2):
    pool_h, pool_w, pool_ch = pool_out_delta.shape
    in_h = np.uint16((pool_h - 1) * stride + size)
    in_w = np.uint16((pool_w - 1) * stride + size)
    in_ch = pool_ch

    pool_out_delta_reshaped = pool_out_delta.transpose(2,0,1)
    pool_out_delta_reshaped = pool_out_delta_reshaped.flatten()
    pool_out_max_location_reshaped = pool_out_max_location.transpose(2,0,1)
    pool_out_max_location_reshaped = pool_out_max_location_reshaped.flatten()

    in_delta_matrix = np.zeros([pool_h*pool_w*pool_ch, size*size])
    in_delta_matrix[np.arange(pool_h*pool_w*pool_ch), 
            pool_out_max_location_reshaped] = pool_out_delta_reshaped
    in_delta = in_delta_matrix.reshape(pool_ch, pool_h, pool_w, size, size)
    in_delta = in_delta.transpose(1, 3, 2, 4, 0)
    in_delta = in_delta.reshape(in_h, in_w, in_ch)
    return in_delta


def rot180(conv_filters):
    rot180_filters = np.zeros((conv_filters.shape))
    for filter_num in range(conv_filters.shape[0]):
        for img_ch in range(conv_filters.shape[-1]):
            rot180_filters[filter_num,:,:,img_ch] = np.flipud(np.fliplr(conv_filters[filter_num,:,:,img_ch]))
    return rot180_filters


def soft_max(z):
    tmp = np.max(z)
    z -= tmp
    z = np.exp(z)
    tmp = np.sum(z)
    z /= tmp

    return z


def add_bias(conv, bias):
    if conv.shape[-1] != bias.shape[0]:
        print("Error Bias Dimension!")
    else:
        for i in range(bias.shape[0]):
            conv[:,:,i] += bias[i,0]
    return conv


def main():
    train_image, train_label, test_image, test_label = dataset_loader()
    net = ConvNet()
    net.SGD(train_image, train_label, test_image, test_label, 50, 100, 3e-5)


if __name__ == '__main__':
    main()
