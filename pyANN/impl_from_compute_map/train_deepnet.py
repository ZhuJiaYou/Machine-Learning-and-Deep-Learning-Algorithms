import numpy as np
import matplotlib.pylab as plt
from data_loader import dataset_loader
from deep_convnet import DeepConvnet
from common.trainer import Trainer


x_train, t_train, x_test, t_test = dataset_loader(
        "/home/zjy/nlp/Machine_Learning_and_Deep_Learning_Algorithms/datasets", flatten=False)
net = DeepConvnet()
trainer = Trainer(net, x_train, t_train, x_test, t_test, epoches=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001}, evaluate_sample_num_per_epoch=1000)
trainer.train()
net.save_params('deep_convnet_params.pkl')
print("Saved Network Parameters!")
