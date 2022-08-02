# # 用于Google Drive的路径处理
# import os
# path = "/content/drive/MyDrive/KDD99"
# os.chdir(path)
# os.listdir(path)

import torch
import datasets
import Net
import trainer
import utils

import torch.nn
import torch.optim
import torch.utils
import logging.config
import torch.utils.data

# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

data='./datasets/kddcup.data_10_percent_corrected'
test = './datasets/test'

def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)

    train_loader = datasets.DataLoader(data)
    train_features, train_labels = train_loader.load(cover=False)
    ## 交叉熵要求标签类型为torch.long
    train_features = torch.from_numpy(train_features.reshape((-1, 1, 11, 11))).type(torch.float32)
    train_labels = torch.from_numpy(train_labels.reshape((-1, ))).type(torch.long)

    test_loader = datasets.DataLoader(test)
    test_features, test_labels = test_loader.load(cover=False)
    test_features = torch.from_numpy(test_features.reshape((-1, 1, 11, 11))).type(torch.float32)
    test_labels = torch.from_numpy(test_labels.reshape((-1, ))).type(torch.long)
    # exit(0)
    device = utils.GPU.try_gpu()
    net = Net.Module()
    net.apply(init_weights)         # 建议初始化，收敛差距极大
    # net.structure()
    
    trainer = trainer.Trainer(
        net, 
        optimizer=torch.optim.SGD(net.parameters(), lr = 0.5),
        loss = torch.nn.CrossEntropyLoss(),
        device=device
    )
    trainer.fit((train_features, train_labels), (test_features, test_labels), batch_size=32)
    trainer.train(epochs=30)
