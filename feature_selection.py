# https://www.programcreek.com/python/example/93974/sklearn.feature_selection.SelectKBest
# https://pypi.org/project/ReliefF/#description


import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import ReliefF


def fit(spam, nonspam, feature, num, alg):
    data = pd.concat([spam, nonspam])
    data = data.iloc[:, feature]
    label = np.zeros(len(data))
    label[0:len(spam)] = 1
    temp:  np.array
    if alg:
        model2 = ReliefF.ReliefF(n_features_to_keep=num, n_neighbors=num)
        model2.fit(np.array(data), np.array(label))
        top = model2.top_features
        temp = np.zeros(len(data.iloc[0]))
        for i in range(len(top)):
            if top[i] < num:
                temp[i] = 1

    else:
        model = SelectKBest(score_func=f_classif, k=num)
        model.fit(data, label)
        temp = model.get_support()
        temp: np.array
        temp = np.array(temp)

    tag = []
    for i in range(len(temp)):
        if temp[i]:
            tag.append(i)

    return data.iloc[:, tag], label, tag
