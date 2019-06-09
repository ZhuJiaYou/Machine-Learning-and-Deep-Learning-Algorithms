import numpy as np
import matplotlib.pylab as plt
from data_loader import dataset_loader
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam


x_train, t_train, x_test, t_test = dataset_loader()

x_train = x_train[:1000]
t_train = t_train[:1000]

max_epoches = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

def __train(weight_init_std):
    bn_net = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10, 
                                 weight_init_std=weight_init_std, use_batchnorm=True)
    net = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10, 
                                 weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch= t_train[batch_mask]

        for _net in (bn_net, net):
            grads = _net.gradient(x_batch, t_batch)
            optimizer.update(_net.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            bn_train_acc = bn_net.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
            print("EPOCH: {0} | NET_ACC({1}) - BN_NET_ACC({2})".format(epoch_cnt, train_acc, bn_train_acc))
            epoch_cnt += 1
            if epoch_cnt >= max_epoches:
                break
    return train_acc_list, bn_train_acc_list

weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epoches)

for i, w in enumerate(weight_scale_list):
    print("=============== {}/16 ==============".format(i+1))
    train_acc_list, bn_train_acc_list = __train(w)
    plt.subplot(4, 4, i+1)
    plt.title("w={}".format(w))

    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Norm', markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', label='Normal', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel('accuracy')
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel('epoches')
    plt.legend(loc='lower right')

plt.show()
