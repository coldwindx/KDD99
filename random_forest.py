import datasets
import logging.config

import sklearn as skl
from sklearn import ensemble

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)

train_data, test_data = datasets.Loader().load()
train_features, train_labels = train_data[:,:-1], train_data[:,-1]
test_features, test_labels = test_data[:,:-1], test_data[:,-1]

# 构造随机森林
clf = skl.ensemble.RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
clf.fit(train_features, train_labels)
# 预测
# predict_result = clf.predict(test_features)
scores = clf.score(test_features, test_labels)
print(scores)
# n_estimators=10, score = 0.9243313281483602
# n_estimators=20, score = 0.9239776611033769
# n_estimators=50, score = 0.925189774521183
# n_estimators=20, score = 0.9256077446652542