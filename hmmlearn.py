import re
import logging
import numpy as np

from typing import DefaultDict
from hmmlearn import hmm

import datasets

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')
# CSIC数据
(train, normal, anomalous) = datasets.Csic().load()
# 词频统计
dict = DefaultDict(int)
for l in train:
    tokens = re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29|/|http://',l)
    for token in tokens:
        dict[token] += 1
# 编码
train_data, train_lens = [], []
for l in train:
    tokens = re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29|/|http://',l)
    for token in tokens:
        train_data.append(dict[token])
    train_lens.append(len(tokens))
train_data = np.array(train_data).reshape((-1, 1))
# 构建模型
remodel = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
remodel.fit(train_data, train_lens)

# 处理测试集
anomalous_data, anomalous_lens = [], []
for l in anomalous:
    tokens = re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29|/|http://',l)
    for token in tokens:
        anomalous_data.append(dict[token])
    anomalous_lens.append(len(tokens))
anomalous_data = np.array(anomalous_data).reshape((-1, 1))
score = remodel.score(anomalous_data, anomalous_lens)
print(score)    # -3744133.893314106
exit(0)

normal_data, normal_lens = [], []
for l in normal:
    tokens = re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29|/|http://',l)
    for token in tokens:
        normal_data.append(dict[token])
    normal_lens.append(len(tokens))
normal_data = np.array(normal_data).reshape((-1, 1))
score = remodel.score(normal_data, normal_lens)
print(score)    # -3761760.932127804

