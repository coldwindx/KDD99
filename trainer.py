import utils
import torch
import logging

from abc import abstractclassmethod
from typing import DefaultDict

log = logging.getLogger('system')

class Plugin():
    AFTER_ONE_BATCH_TRAIN   = 1
    AFTER_ONE_EPOCH_TRAIN   = 2
    AFTER_ONE_BATCH_TEST    = 4
    AFTER_ONE_EPOCH_TEST    = 8
    AFTER_ONE_EPOCH         = 16
    BEFORE_RETURN           = 32
    ALL                     = 63

    @abstractclassmethod
    def position(self) -> int:
        pass
    def after_one_batch_train(self):
        raise NotImplementedError
    def after_one_epoch_train(self):
        raise NotImplementedError
    def after_one_batch_test(self):
        raise NotImplementedError
    def after_one_epoch_test(self):
        raise NotImplementedError
    def after_one_epoch(self):
        raise NotImplementedError
    def before_return(self):
        raise NotImplementedError

class AccuractPlugin(Plugin):
    def __init__(self, metrics) -> None:
        super().__init__()
        metrics['train_accuracy_sum'] = 0.0
        metrics['train_data_size'] = 0  
        metrics['train_loss_sum'] = 0.0
        metrics['train_data_size_sum'] = 0
        metrics['test_accuracy_sum'] = 0.0
        metrics['test_data_size'] = 0

    def position(self) -> int:
        return  Plugin.ALL
    def after_one_batch_train(self, metrics, o, y, loss):
        metrics['train_accuracy_sum'] += utils.accuracy(o, y)
        metrics['train_data_size'] += y.numel()
        metrics['train_data_size_sum'] += y.numel()
        metrics['train_loss_sum'] += loss

    def after_one_epoch_train(self, metrics):
        metrics['train_accuracy'] = metrics['train_accuracy_sum'] / metrics['train_data_size']
        metrics['train_loss'] = metrics['train_loss_sum'] / metrics['train_data_size']
        metrics['train_accuracy_sum'] = metrics['train_data_size'] = metrics['train_loss_sum'] = 0
    def after_one_batch_test(self, metrics, o, y):
        metrics['test_accuracy_sum'] += utils.accuracy(o, y)
        metrics['test_data_size'] += y.numel()
    def after_one_epoch_test(self, metrics):
        metrics['test_accuracy'] = metrics['test_accuracy_sum'] / metrics['test_data_size']
        metrics['test_accuracy_sum'] = metrics['test_data_size'] = 0
    def after_one_epoch(self, metrics, epoch):
        log.info(f'epoch: {epoch}, \
                train_loss is {metrics["train_loss"]:.3f}, \
                train_accuracy is {metrics["train_accuracy"]:.3f}, \
                test_accuracy is {metrics["test_accuracy"]:.3f}')
    def before_return(self, metrics, timer, device):
        log.info(f'{metrics["train_data_size_sum"] / timer.sum():.1f} examples/sec 'f'on {str(device)}')

class Trainer:
    def __init__(self, net, optimizer, loss, device = 'cpu') -> None:
        self.net = net.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.iterations = 0
        self.metrics = DefaultDict(str)
        self.plugins = [AccuractPlugin(self.metrics)]

    def fit(self, data, test, batch_size = 64):
        (train_features, train_labels) = data
        data = torch.utils.data.TensorDataset(train_features, train_labels)
        self.data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

        (test_features, test_labels) = test
        test = torch.utils.data.TensorDataset(test_features, test_labels)
        self.test = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    def train(self, epochs = 1):
        timer = utils.Timer()
        for i in range(1, epochs + 1):
            timer.start()
            self.net.train()
            self._train()
            timer.stop()
            self.net.eval()
            self._test()
            for plugin in self.plugins:
                if (plugin.position() & Plugin.AFTER_ONE_EPOCH):
                    plugin.after_one_epoch(self.metrics, i)
        for plugin in self.plugins:
            if (plugin.position() & Plugin.BEFORE_RETURN):
                plugin.before_return(self.metrics, timer, self.device)
    
    def _train(self):
        for (X, y) in self.data:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            o = self.net(X)
            l = self.loss(o, y)
            l.backward()
            self.optimizer.step()
            with torch.no_grad():
                for plugin in self.plugins:
                    if (plugin.position() & Plugin.AFTER_ONE_BATCH_TRAIN):
                        plugin.after_one_batch_train(self.metrics, o, y, l * X.shape[0])
        for plugin in self.plugins:
            if (plugin.position() & Plugin.AFTER_ONE_EPOCH_TRAIN):
                plugin.after_one_epoch_train(self.metrics)
        self.iterations += 1
    
    @torch.no_grad()
    def _test(self):
        for (X, y) in self.test:
            X, y = X.to(self.device), y.to(self.device)
            o = self.net(X)
            for plugin in self.plugins:
                if (plugin.position() & Plugin.AFTER_ONE_BATCH_TEST):
                    plugin.after_one_batch_test(self.metrics, o, y)
        for plugin in self.plugins:
            if (plugin.position() & Plugin.AFTER_ONE_EPOCH_TEST):
                plugin.after_one_epoch_test(self.metrics)