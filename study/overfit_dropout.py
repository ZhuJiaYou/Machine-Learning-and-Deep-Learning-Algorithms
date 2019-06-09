import numpy as np
import matplotlib.pylab as plt
from data_loader import dataset_loader
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer


x_train, t_train, x_test, t_test = dataset_loader()

x_train = x_train[:300]
t_train = t_train[:300]

use_dropout = True
dropout_ratio = 0.2

net = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, 
                          use_dropout=use_dropout, dropout_ratio=dropout_ratio)
trainer = Trainer(net, x_train, t_train, x_test, t_test, epoches=301, mini_batch_size=100, optimizer='SGD', 
                  optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(len(train_acc_list))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker=markers['train'], label='train', markevery=10)
plt.plot(x, test_acc_list, marker=markers['test'], label='test', markevery=10)
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
