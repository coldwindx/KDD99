import logging
import datasets

from logging import config
from sklearn import svm

TRAIN_DATA_PATH = './datasets/kddcup.data_10_percent_corrected'
TEST_DDATA_PATH = './datasets/test'

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')

train_features, train_labels = datasets.DataLoader(TRAIN_DATA_PATH).load(cover=False)
test_features, test_labels = datasets.DataLoader(TEST_DDATA_PATH).load(cover=False)

clf = svm.SVC(gamma='scale')
clf.fit(train_features, train_labels)
score = clf.score(test_features, test_labels)
log.info(f'The accuracy using SVM is {score}')