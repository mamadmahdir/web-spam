import numpy as np
import pandas as pd


def gauss_mf(x: int, c: int, a: int):
    # c is mean
    # a is std
    return np.e ** ((-(1 / 2)) * (((x - c) / a) ** 2))


def fit(minority: pd.DataFrame, majority: pd.DataFrame, feature: list, alpha=0.5):
    c_min = minority.iloc[:, feature].mean()
    a_min = minority.iloc[:, feature].std()

    c_maj = majority.iloc[:, feature].mean()
    a_maj = majority.iloc[:, feature].std()

    def temp(s: pd.Series):
        s = np.array(s)
        fval = 0
        for j in range(len(s)):
            maj_u = gauss_mf(s[j], c_maj[j], a_maj[j])
            min_u = gauss_mf(s[j], c_min[j], a_min[j])
            fval += min_u + (1 - maj_u)

        return fval

    score = pd.Series(majority.iloc[:, feature].apply(temp, axis=1))
    num_remove = int(len(majority) - ((1-alpha)/alpha) * len(minority))

    majority = majority.drop(score.nlargest(num_remove).index, axis=0)

    return minority, majority
