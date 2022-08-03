# # 用于Google Drive的路径处理
# import os
# path = "/content/drive/MyDrive/KDD99"
# os.chdir(path)
# os.listdir(path)

import os
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

if __name__ == '__main__':
    logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)

    train_features, train_labels = datasets.DataLoader(data).load(cover=True)
    ## 交叉熵要求标签类型为torch.long
    train_features = torch.from_numpy(train_features.reshape((-1, 1, 11, 11))).type(torch.float32)
    train_labels = torch.from_numpy(train_labels.reshape((-1, ))).type(torch.long)

    test_features, test_labels = datasets.DataLoader(test).load(cover=True)
    test_features = torch.from_numpy(test_features.reshape((-1, 1, 11, 11))).type(torch.float32)
    test_labels = torch.from_numpy(test_labels.reshape((-1, ))).type(torch.long)
    # exit(0)
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
    trainer.fit((train_features, train_labels), (test_features, test_labels), batch_size=32)
    trainer.train(epochs=30)

    torch.save(net.state_dict(), './module/conv.mod')