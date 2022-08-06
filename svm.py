# # 用于Google Drive的路径处理
# import os
# path = "/content/drive/MyDrive/KDD99"
# os.chdir(path)
# os.listdir(path)

import logging
import numpy as np
import datasets

from logging import config
from sklearn import svm

TRAIN_DATA_PATH = './datasets/kddcup.data_10_percent_corrected'
TEST_DDATA_PATH = './datasets/test'

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')

train_data, test_data = datasets.Loader().load()
train_features, train_labels = train_data[:,:-1], train_data[:,-1]
test_features, test_labels = test_data[:,:-1], test_data[:,-1]
## 归一化
dmax, dmin = np.max(train_features, axis=0), np.min(train_features, axis=0)
dmax, dmin = dmax.reshape((1, -1)), dmin.reshape((1, -1))
train_features = (train_features - dmin) / (dmax - dmin + 1e-8)
test_features = (test_features - dmin) / (dmax - dmin + 1e-8)


clf = svm.SVC(gamma='scale')
clf.fit(train_features, train_labels)
score = clf.score(test_features, test_labels)
# 0.9250418773932809，运行时间1h2m
log.info(f'The accuracy using SVM is {score}')