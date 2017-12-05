import numpy as np
from random import sample
import matplotlib.pyplot as plt


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
