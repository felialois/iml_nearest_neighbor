import kNNAlgorithm
import sklearn.feature_selection as sk
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np


def calculate_weights(data, target):
    """

    :param data:
    :param target:
    :return:
    """
    # fit an Extra Trees model to the data
    model = ExtraTreesClassifier()
    model.fit(data, target)
    # display the relative importance of each attribute
    return model.feature_importances_
