from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from skrebate import ReliefF
import utils


def calculate_weights(data, target, algorithm):
    """

    :param data:
    :param target:
    :return:
    """
    target = utils.encode_target(target)
    if algorithm == 'TreeClassifier':
        # fit an Extra Trees model to the data
        model = ExtraTreesClassifier()
        model.fit(data, target)
        # display the relative importance of each attribute
        return model.feature_importances_
    if algorithm == 'Relieff':
        fs = ReliefF()
        fs.fit(data, target)
        feature_sum = np.sum(fs.feature_importances_)
        return [fs.feature_importances_[f] / feature_sum for f in range(len(fs.feature_importances_))]
