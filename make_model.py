# https://scikit-learn.org/stable/modules/tree.html

import sklearn.tree as tree


def fit(data, label):
    model = tree.DecisionTreeClassifier()
    model.fit(data, label)
    return model
