import sklearn
import logging
import datasets

from sklearn import neighbors

logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')

train_data, test_data = datasets.Loader().load()
train_features, train_labels = train_data[:,:-1], train_data[:,-1]
test_features, test_labels = test_data[:,:-1], test_data[:,-1]

num_neighbors = [5, 10, 15, 20, 25]
for k in num_neighbors:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)
    score = knn.score(test_features, test_labels)
    log.info(f'K Nearest Neighbors score is {score} when neighbors is {k}')