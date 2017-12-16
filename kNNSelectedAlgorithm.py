from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
import numpy as np


def remove_features(data, target, fn):
    """

    :param target:
    :param fn:
    :param k:
    :param data:
    :return:
    """
    selected_data = []
    if fn == 'variance':
        sel = VarianceThreshold(threshold=(.1 * (1 - .8)))
        selected_data = sel.fit_transform(data)
    elif fn == 'L1':
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, target)
        model = SelectFromModel(lsvc, prefit=True)
        selected_data = model.transform(data)

    selected_t = np.transpose(selected_data)
    data_t = np.transpose(data)

    i = 0
    removed_cols = []
    for i, col in enumerate(data_t):
        if col not in selected_t:
            removed_cols.append(i)
    return selected_data, removed_cols
