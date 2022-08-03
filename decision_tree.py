import datasets
import logging.config

import pydotplus
from sklearn import tree

data='./datasets/kddcup.data_10_percent_corrected'
test = './datasets/test'

if __name__ == '__main__':
    logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)

    train_loader = datasets.OneHotLoader(test)
    train_features, train_labels = train_loader.load(cover=False)

    test_loader = datasets.OneHotLoader(test)
    test_features, test_labels = test_loader.load(cover=False)

    # 构造一棵多分类决策树
    ## 限制一下树的结点，预测准确率下降不到0.02
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=20)
    clf = clf.fit(train_features, train_labels)
    # 预测
    # predict_result = clf.predict(test_features)
    scores = clf.score(test_features, test_labels)
    print(scores)
    # 导出决策树结构
    dot = tree.export_graphviz(clf, out_file=None)  # 默认导出.dot文件
    graph = pydotplus.graph_from_dot_data(dot)
    graph.write_pdf('./module/decision.pdf')


