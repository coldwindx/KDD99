from torch import device
import torch.nn

class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, 16, kernel_size=4, padding=1)
        self.h1 = torch.nn.ReLU()
        self.p1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.h2 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.a2 = torch.nn.ReLU()
        self.f  = torch.nn.Flatten()
        self.l1 = torch.nn.Linear(200, 100)
        self.h3 = torch.nn.ReLU()
        self.d1 = torch.nn.Dropout(0.2)
        self.l2 = torch.nn.Linear(100, 40)
        self.h4 = torch.nn.ReLU()
        self.d2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(40, 5)
        self.h5 = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.c1(x)
        x = self.h1(x)
        x = self.p1(x)
        x = self.h2(x)
        x = self.a2(x)
        x = self.f (x)
        x = self.l1(x)
        x = self.h3(x)
        x = self.d1(x)
        x = self.l2(x)
        x = self.h4(x)
        x = self.d2(x)
        x = self.l3(x)
        x = self.h5(x)
        return x

    def structure(self):
        X = torch.rand(size=(1, 1, 11, 11), dtype=torch.float32)
        for layer in self.named_children():
            X = layer[1](X)
            print(layer[1].__class__.__name__,'output shape: \t',X.shape)