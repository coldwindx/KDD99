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

from torch.nn import functional as F

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(121, 64)
        torch.nn.init.xavier_normal_(self.l1.weight)
        self.l2 = torch.nn.Linear(64, 5)
        torch.nn.init.xavier_normal_(self.l2.weight)
    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        x = F.softmax(self.l2(x))
        return x

train_data, test_data = datasets.Loader().load()
train_features, train_labels = train_data[:,:-1], train_data[:,-1]
test_features, test_labels = test_data[:,:-1], test_data[:,-1]
## 归一化
dmax, dmin = np.max(train_features, axis=0), np.min(train_features, axis=0)
dmax, dmin = dmax.reshape((1, -1)), dmin.reshape((1, -1))
train_features = (train_features - dmin) / (dmax - dmin + 1e-8)
test_features = (test_features - dmin) / (dmax - dmin + 1e-8)
## 交叉熵要求标签类型为torch.long
train_features = torch.from_numpy(train_features).type(torch.float32)
train_labels = torch.from_numpy(train_labels).type(torch.long)
test_features = torch.from_numpy(test_features).type(torch.float32)
test_labels = torch.from_numpy(test_labels).type(torch.long)


device = utils.GPU.try_gpu()
net = Net()
if os.path.exists('./module/softmax.mod'):
    net.load_state_dict(torch.load('./module/softmax.mod'))
trainer = trainer.Trainer(
    net, 
    optimizer=torch.optim.SGD(net.parameters(), lr = 0.2),
    loss = torch.nn.CrossEntropyLoss(),
    device=device
)
trainer.fit((train_features, train_labels), (test_features, test_labels), batch_size=128)
trainer.train(epochs=30)
torch.save(net.state_dict(), './module/softmax.mod')