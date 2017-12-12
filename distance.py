import numpy as np


def distance(tp, x, y, weights):
    """

    :param tp: Hamming, Euclidean, Cosine and Canberra
    :param x:
    :param y:
    :param weights:
    :return:
    """

    for index, d in enumerate(zip(x, y)):
        if type(d[0]) is str or type(d[1]) is str:
            if d[0] == d[1]:
                x[index] = 1
                y[index] = 1
            else:
                x[index] = 1
                y[index] = 0

    if tp == 'Hamming':
        hamming = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                hamming += (1 * weights[i])
        return hamming
    elif tp == 'Euclidean':
        a = zip(x, y)
        return np.sqrt(np.sum(((x[i] - y[i]) ** 2) * weights[i] for i in range(len(x))))
    elif tp == 'Cosine':
        numerator = np.sum([(x[i] * y[i]) * weights[i] for i in range(len(x))])
        denominator = np.sqrt(np.sum([j ** 2 for j in x])) * np.sqrt(np.sum([j ** 2 for j in y]))
        return 1 - (numerator / denominator)
    elif tp == 'Canberra':
        return np.sum(
            [(np.abs(x[i] - y[i]) * weights[i]) / ((np.abs(x[i]) + np.abs(y[i])) * weights[i]) for i in range(len(x))])
    else:
        raise ValueError
