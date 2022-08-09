import sklearn
import logging
import datasets

from sklearn import naive_bayes

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')

train_data, test_data = datasets.Loader().load()
train_features, train_labels = train_data[:,:-1], train_data[:,-1]
test_features, test_labels = test_data[:,:-1], test_data[:,-1]
# 构建高斯贝叶斯
gnb = naive_bayes.GaussianNB()
gnb.fit(train_features, train_labels)
# 测试
test_score = gnb.score(test_features, test_labels)
log.info(f'Gaussian Naive Bayes, test score: {test_score}')
# Gaussian Naive Bayes, test score: 0.8514984229664949