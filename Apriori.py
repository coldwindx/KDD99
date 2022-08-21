import re
import logging
import efficient_apriori as eapriori
import pyfpgrowth

import datasets

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')
# CSIC数据
train = datasets.Csic().load()
# 单词向量
train_data = []
for l in train:
    tokens = re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29|/|http://',l)
    train_data.append(tokens)

patterns = pyfpgrowth.find_frequent_patterns(train_data[:10000], 3)
rules = pyfpgrowth.generate_association_rules(patterns, 0.99)
log.info('!---------------------- pyfpgrowth ----------------------------')
log.info(rules)
exit(0)
# 关联分析
itemsets, rules = eapriori.apriori(train_data, min_support = 0.15, min_confidence = 0.99)
log.info(rules)
