from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris


def remove_features(data, target, function):
    """

    :param target:
    :param function:
    :param k:
    :param data:
    :return:
    """
    if function == 'variance':
        sel = VarianceThreshold(threshold=(.1 * (1 - .8)))
        return sel.fit_transform(data)
    elif function == 'L1':
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, target)
        model = SelectFromModel(lsvc, prefit=True)
        return model.transform(data)

