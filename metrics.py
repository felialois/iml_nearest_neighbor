import numpy
from sklearn.metrics import precision_score, recall_score


def accuracy(x_true, x_predicted, average):
    """

    :param x_true:
    :param x_predicted:
    :param average:
    :return:
    """
    return precision_score(x_true, x_predicted, average=average)


def recall(x_true, x_predicted):
    """

    :param x_true:
    :param x_predicted:
    :return:
    """
    return recall_score(x_true, x_predicted, average='macro')

