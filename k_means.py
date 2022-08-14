import sklearn
import logging
import datasets

from sklearn import cluster
# 日志
logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')
# 数据
train_data, test_data = datasets.Loader().load()
train_features, train_labels = train_data[:,:-1], train_data[:,-1]
test_features, test_labels = test_data[:,:-1], test_data[:,-1]
# K-Means
kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(train_features)
centroids = kmeans.cluster_centers_
print(centroids)