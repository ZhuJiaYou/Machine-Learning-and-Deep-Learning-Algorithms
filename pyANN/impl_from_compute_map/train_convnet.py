import numpy as np
import matplotlib.pylab as plt
from data_loader import dataset_loader
from simple_convnet import SimpleConvnet
from common.trainer import Trainer


x_train, t_train, x_test, t_test = dataset_loader(
        "/home/zjy/nlp/Machine_Learning_and_Deep_Learning_Algorithms/datasets", flatten=False)

# x_train = x_train[:500]
# t_train = t_train[:500]

max_epoches = 20
net = SimpleConvnet(input_dim=(1, 28, 28),
                    conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                    hidden_size=100, output_size=10, weight_init_std=0.01)
trainer = Trainer(net, x_train, t_train, x_test, t_test, epoches=max_epoches, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001}, evaluate_sample_num_per_epoch=1000)
trainer.train()
net.save_params('params.pkl')
print("Saved Network Parameters!")

markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epoches)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
