import numpy as np
import matplotlib.pylab as plt
from data_loader import dataset_loader
from common.util import shuffle_dataset
from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer


x_train, t_train, x_test, t_test = dataset_loader()

x_train = x_train[:500]
t_train = t_train[:500]

validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)

x_valuation = x_train[:validation_num]
t_valuation = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

def __train(lr, weight_decay, epoches=50):

    net = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], 
                              output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(net, x_train, t_train, x_valuation, t_valuation, epoches=epoches, mini_batch_size=100, 
                      optimizer='SGD', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()
    return trainer.train_acc_list, trainer.test_acc_list


optimization_trial = 100
results_validation = {}
results_train = {}
for _ in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    train_acc_list, validation_acc_list = __train(lr, weight_decay)
    print("VALIDATION ACC: {0} | LR: {1} | WEIGHT DECAY: {2}".format(validation_acc_list[-1], lr, 
                                                                     weight_decay))
    key = "lr:{0}, weight decay:{1}".format(lr, weight_decay)
    results_validation[key] = validation_acc_list
    results_train[key] = train_acc_list

print("================================= HIPER-PARAMETER OPTIMIZATION RESULT " + 
      "=========================================")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, validation_acc_list in sorted(results_validation.items(), key=lambda x: x[1][-1], reverse=True):
    print("BEST-{0}(VALIDATION ACC:{1}) | {2}".format(i+1, validation_acc_list[-1], key))
    plt.subplot(row_num, col_num, i+1)
    plt.title("BEST-{}".format(i+1))
    plt.ylim(0, 1.0)
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(validation_acc_list))
    plt.plot(x, validation_acc_list)
    plt.plot(x, results_train[key], '--')
    i += 1
    if i >= graph_draw_num:
        break
plt.show()
