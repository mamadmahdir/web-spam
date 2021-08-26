import numpy as np
import pandas as pd

import read_clean as rc
import undersampling as uds
import feature_selection as fs
import make_model as mm

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

alg = 1
f1 = (0.09, 34, 1.75)
f2 = (0.4, 40, 0.75)
f3 = (0.1, 2,  0.25)


def feature_list(all_data: pd.DataFrame):
    return np.array(range((len(all_data.columns))))[2:]


def give_label(name, alpha=0.3, n_feature=0):
    data_raw = pd.read_csv(name)
    feature = feature_list(data_raw)

    temp = list(rc.fit(data_raw))

    temp[0], temp[1] = uds.fit(temp[0], temp[1], feature, alpha)

    data, label, tag = fs.fit(temp[0], temp[1], feature, n_feature, alg)
    model = mm.fit(data, label)

    new_label = pd.Series(model.predict(data_raw.iloc[:, feature[tag]]))
    ids = data_raw['#hostid']

    return pd.Series(new_label, index=ids)


def ensemble(all_label: pd.DataFrame) -> np.array:
    return all_label.sum(axis=1) >= 1.5


def splitter(data: pd.DataFrame, label: pd.DataFrame):
    temp1 = data[label.iloc[data["#hostid"]].values]
    temp2 = data[~label.iloc[data["#hostid"]].values]
    return temp1, temp2


print("#.", end=" ")
first = give_label("uk-2007-05.content_based_features.csv", f1[0], f1[1]) * f1[2]

print("#.", end=" ")
second = give_label("uk-2007-05.link_based_features.csv", f2[0], f2[1]) * f2[2]

print("#.", end=" ")
third = give_label("uk-2007-05.link_based_features_transformed.csv", f3[0], f3[1]) * f3[2]

final_label = ensemble(pd.concat([first, second, third], axis=1))
test_label = pd.read_csv('WEBSPAM-UK2007-SET2-labels.txt', sep=' ', header=None).iloc[:, [0, 1]]


def give_label_test(name, alpha=0.3, n_feature=0):
    raw_data = pd.read_csv(name)
    feature = feature_list(raw_data)
    min_d, maj = splitter(raw_data, final_label)
    min_d, maj = uds.fit(min_d, maj, feature, alpha)
    print("#.", end=" ")
    data, label, tag = fs.fit(min_d, maj, feature, n_feature, alg)
    print("#.", end=" ")
    model = mm.fit(data, label)
    print("#.", end=" ")
    test_data = raw_data[raw_data["#hostid"].isin(np.array(test_label.iloc[:, 0]))]
    return pd.Series(model.predict(test_data.iloc[:, feature[tag]]))


print("#.", end=" ")
first_s = give_label_test("uk-2007-05.content_based_features.csv", f1[0], f1[1]) * f1[2]

print("#.", end=" ")
second_s = give_label_test("uk-2007-05.link_based_features.csv", f2[0], f2[1]) * f2[2]

print("#.", end=" ")
third_s = give_label_test("uk-2007-05.link_based_features_transformed.csv", f3[0], f3[1]) * f3[2]

final_label_s = ensemble(pd.concat([first_s, second_s, third_s], axis=1))

l1 = final_label_s
l2 = test_label.iloc[:, 1]

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(l2)):
    if (l1[i] == 1) and ('spam' == l2[i]):
        TP += 1

    if (l1[i] == 1) and ('nonspam' == l2[i]):
        FN += 1

    if (l1[i] == 0) and ('spam' == l2[i]):
        FP += 1

    if (l1[i] == 0) and ('nonspam' == l2[i]):
        TN += 1

print()
print(TP, FP)
print(TN, FN)

l2 = l2 == 'spam'

print("f_score:", f1_score(l2, l1,  zero_division=1, average='binary'))
print("accuracy:", accuracy_score(l2, l1))
print("precision:", TP / (TP + FP))
print("Recall:", TP / (TP + FN))
print("specificity:", TN / (TN + FP))
print("AUC:", (1/2) * (TN / (TN + FP) + TP / (TP + FN)))
