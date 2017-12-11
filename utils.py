import numpy as np
from random import sample
import matplotlib.pyplot as plt
import math


def normalize_columns(data):
    """

    :param data:
    :return: data with all the columns normalized
    """
    norm_data = []
    for col in data:
        if type(col[0]) is str:
            norm_data.append(col)
        else:
            min_v = min(col)
            max_v = max(col)
            norm_col = []
            for element in col:
                norm_col.append((element - min_v) / (max_v - min_v))
            norm_data.append(norm_col)
    return norm_data


def encode_data(data):
    """
    Transform all the categorical columns into encoded numeric columns
    :param data: dataset
    :return: dataset with all the categorical columns replaces
    """
    d = np.transpose(data)
    res = []
    for col in d:
        if type(col[0]) is str:
            unique_values = np.unique(col)
            new_col = []
            for i in range(len(col)):
                new_col.append(np.where(unique_values == col[i])[0][0])
            res.append(new_col)
        else:
            res.append([-1 if math.isnan(x) else x for x in col])
    return np.transpose(res)

