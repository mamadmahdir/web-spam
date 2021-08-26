import numpy as np
import pandas as pd


def fit(data: pd.DataFrame):
    train_label = pd.read_csv('WEBSPAM-UK2007-SET1-labels.txt', sep=' ', header=None).iloc[:, [0, 1]]
    test_label = pd.read_csv('WEBSPAM-UK2007-SET2-labels.txt', sep=' ', header=None).iloc[:, [0, 1]]

    train_data_spam = data[data["#hostid"].isin(np.array(train_label[train_label.iloc[:, 1] == 'spam'].iloc[:, 0]))]
    train_data_non = data[data["#hostid"].isin(np.array(train_label[train_label.iloc[:, 1] == 'nonspam'].iloc[:, 0]))]

    test_data_spam = data[data["#hostid"].isin(np.array(test_label[test_label.iloc[:, 1] == 'spam'].iloc[:, 0]))]
    test_data_non = data[data["#hostid"].isin(np.array(test_label[test_label.iloc[:, 1] == 'nonspam'].iloc[:, 0]))]

    data = data.drop(np.array(train_data_spam.index), axis=0)
    data = data.drop(np.array(train_data_non.index), axis=0)
    data = data.drop(np.array(test_data_spam.index), axis=0)
    data = data.drop(np.array(test_data_non.index), axis=0)

    return train_data_spam, train_data_non, test_data_spam, test_data_non, data
