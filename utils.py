import torch
import time
import numpy as np

def accuracy(o, y):
    '''计算准确率'''
    m, n = o.shape
    if 1 < m and 1 < n:
        o = o.argmax(axis = 1)
    cmp = (o.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

class GPU:
    @staticmethod
    def try_gpu(i = 0):
        '''如果存在，则返回gpu(i)，否则返回cpu()'''
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f"cuda:{i}")
        return torch.device('cpu')
    @staticmethod
    def try_all_gpu():
        '''返回所有可用的GPU，如果没有GPU，则返回[cpu(),]'''
        devices = [ torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        return devices if devices else [torch.device('cpu')]

class Timer:
    '''记录运行时间'''
    def __init__(self) -> None:
        self.times = []
        self.tik = time.time()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def sum(self):
        return sum(self.times)
    def avg(self):
        return sum(self.times) / len(self.times)
    def cunsum(self):
        return np.array(self.times).cumsum().tolist()


