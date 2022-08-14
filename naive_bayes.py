import sklearn
import logging
import datasets

from sklearn import naive_bayes

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')

train_data, test_data = datasets.Loader().load()
train_features, train_labels = train_data[:,:-1], train_data[:,-1]
test_features, test_labels = test_data[:,:-1], test_data[:,-1]
# 补充朴素贝叶斯
alphas = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
for a in alphas:
    cnb = naive_bayes.ComplementNB(alpha=a)
    cnb.fit(train_features, train_labels)
    cnb_score = cnb.score(test_features, test_labels)
    log.info(f'Complement Naive Bayes, test score: {cnb_score} when alpha is {a}')
exit(0)
# 构建高斯贝叶斯
gnb = naive_bayes.GaussianNB()
gnb.fit(train_features, train_labels)
# 测试
gnb_score = gnb.score(test_features, test_labels)
log.info(f'Gaussian Naive Bayes, test score: {gnb_score}')
# Gaussian Naive Bayes, test score: 0.8514984229664949

# 构建多项分布朴素贝叶斯
for a in alphas:
    mnb = naive_bayes.MultinomialNB(alpha=a)
    mnb.fit(train_features, train_labels)
    # 测试
    mnb_score = mnb.score(test_features, test_labels)
    log.info(f'Multinomial Naive Bayes, test score: {mnb_score} when alpha is {a}')
    # Multinomial Naive Bayes, test score: 0.8375092837599308
