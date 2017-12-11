from sklearn.feature_selection import VarianceThreshold


def remove_features(data, target, function, k=0):
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