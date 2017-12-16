import numpy
from sklearn.metrics import precision_score, recall_score
from scipy.stats import wilcoxon


def accuracy(x_true, x_predicted):
    """

    :param x_true:
    :param x_predicted:
    :param average:
    :return:
    """
    return precision_score(x_true, x_predicted, average='micro')


def recall(x_true, x_predicted):
    """

    :param x_true:
    :param x_predicted:
    :return:
    """
    return recall_score(x_true, x_predicted, average='macro')


def wilcoxon_analyze(x, y):
    """

    :param x: First array of results
    :param y: Second array of results
    :return:
    """
    return wilcoxon(x, y)
