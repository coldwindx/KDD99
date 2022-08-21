# # 用于Google Drive的路径处理
# import os
# path = "/content/drive/MyDrive/KDD99"
# os.chdir(path)
# os.listdir(path)

import logging
import numpy as np
import datasets
import trainer as T
import utils

from logging import config
from sklearn import decomposition

import torch.nn
import torch.optim
import torch.utils
import logging.config
import torch.utils.data

from torch.nn import functional as F

TRAIN_DATA_PATH = './datasets/kddcup.data_10_percent_corrected'
TEST_DDATA_PATH = './datasets/test'

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')


class Net(torch.nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.l1 = torch.nn.Linear(inputs, 256)
        torch.nn.init.xavier_normal_(self.l1.weight)
        self.l2 = torch.nn.Linear(256, 5)
        torch.nn.init.xavier_normal_(self.l2.weight)
    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return x

train_data, test_data = datasets.Loader().load()
train_features, train_labels = train_data[:,:-1], train_data[:,-1]
test_features, test_labels = test_data[:,:-1], test_data[:,-1]
train_len, test_len = train_features.shape[0], test_features[0]

train_labels = torch.from_numpy(train_labels).type(torch.long)
test_labels = torch.from_numpy(test_labels).type(torch.long)
# 主成分分析

for i in range(2, 121, 2): 
    pca = decomposition.PCA(n_components=i, whiten=True)
    pca.fit(train_features)
    train_x = pca.transform(train_features)
    test_x = pca.transform(test_features)
    log.info(f'current dimension is {i}')

    train_x = torch.from_numpy(train_x).type(torch.float32)
    test_x = torch.from_numpy(test_x).type(torch.float32)

    device = utils.GPU.try_gpu()
    net = Net(i)
    trainer = T.Trainer(
        net, 
        optimizer=torch.optim.SGD(net.parameters(), lr = 0.2),
        loss = torch.nn.CrossEntropyLoss(),
        device=device
    )
    trainer.fit((train_x, train_labels), (test_x, test_labels), batch_size=128)
    trainer.train(epochs=5)

