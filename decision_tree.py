import datasets
import logging.config

import pydotplus
from sklearn import tree

if __name__ == '__main__':
    logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)

    train_data, test_data = datasets.Loader().load()
    train_features, train_labels = train_data[:,:-1], train_data[:,-1]
    test_features, test_labels = test_data[:,:-1], test_data[:,-1]

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
