import torch
import torchvision
import trainer
import requests

from torch import nn
from torchvision import transforms

from utils import GPU

path = './datasets/'
trans = [transforms.ToTensor()]
trans = transforms.Compose(trans)
# 下载数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root = path, train=True, download=True,
    transform=trans     # 自动转为torch张量
)
mnist_test = torchvision.datasets.FashionMNIST(
    root = path, train=False, download=True,
    transform=trans     # 自动转为torch张量
)
device = GPU.try_gpu()
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
    
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)

net = net.to(device)
trainer = trainer.Trainer(
    net, 
    optimizer=torch.optim.SGD(net.parameters(), lr=0.9),
    loss = nn.CrossEntropyLoss(),
    device=device
)
trainer.fit(mnist_train, mnist_test, batch_size=256)
trainer.train(epochs=10)