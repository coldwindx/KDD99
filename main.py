# # 用于Google Drive的路径处理
# import os
# path = "/content/drive/MyDrive/KDD99"
# os.chdir(path)
# os.listdir(path)

import os
import utils
import torch
import numpy as np
import Net
import trainer
import datasets


import torch.nn
import torch.optim
import torch.utils
import logging.config
import torch.utils.data

# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
from matplotlib import pyplot as plt
from trainer import Plugin
# 动态绘图的Demo
class AccuracyCurvePlugin(Plugin):
    def __init__(self, metrics) -> None:
        super().__init__()
        metrics['train_accuracy_y'] = []
        metrics['train_accuracy_x'] = [] 
        metrics['current_epoch'] = 0

    def position(self) -> int:
        return  Plugin.AFTER_ONE_BATCH_TRAIN | Plugin.AFTER_ONE_EPOCH_TRAIN | Plugin.BEFORE_RETURN
    def after_one_batch_train(self, metrics, o, y, loss, batch):
        metrics['train_accuracy_y'].append(utils.accuracy(o, y) / y.numel())
        metrics['train_accuracy_x'].append(metrics['current_epoch'] + batch / metrics['train_batchs'])
        plt.plot(metrics['train_accuracy_x'], metrics['train_accuracy_y'])
        plt.pause(1.0)
    def after_one_epoch_train(self, metrics):
        metrics['current_epoch'] += 1

    def before_return(self, metrics, timer, device):
        plt.show()

if __name__ == '__main__':
    logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)

    train_data, test_data = datasets.Loader().load()
    train_features, train_labels = train_data[:,:-1], train_data[:,-1]
    test_features, test_labels = test_data[:,:-1], test_data[:,-1]
    ## 归一化
    dmax, dmin = np.max(train_features, axis=0), np.min(train_features, axis=0)
    dmax, dmin = dmax.reshape((1, -1)), dmin.reshape((1, -1))
    train_features = (train_features - dmin) / (dmax - dmin + 1e-8)
    test_features = (test_features - dmin) / (dmax - dmin + 1e-8)
    ## 交叉熵要求标签类型为torch.long
    train_features = torch.from_numpy(train_features.reshape((-1, 1, 11, 11))).type(torch.float32)
    train_labels = torch.from_numpy(train_labels.reshape((-1, ))).type(torch.long)
    test_features = torch.from_numpy(test_features.reshape((-1, 1, 11, 11))).type(torch.float32)
    test_labels = torch.from_numpy(test_labels.reshape((-1, ))).type(torch.long)

    device = utils.GPU.try_gpu()
    net = Net.Module()
    if os.path.exists('./module/conv.mod'):
        net.load_state_dict(torch.load('./module/conv.mod'))

    trainer = trainer.Trainer(
        net, 
        optimizer=torch.optim.SGD(net.parameters(), lr = 0.5),
        loss = torch.nn.CrossEntropyLoss(),
        device=device
    )
    trainer.fit((train_features, train_labels), (test_features, test_labels), batch_size=128)

    # trainer.register(AccuracyCurvePlugin(trainer.metrics))
    trainer.train(epochs=3)

    torch.save(net.state_dict(), './module/conv.mod')