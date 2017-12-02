import utils
import numpy as np
import distance


def nearest_neighbor(data_train, data_test, k):
    """

    :param data_train: (data, class)
    :param data_test:  (data)
    :param k:
    :return:
    """
    cls = []
    for y in data_test:
        # We alculate all the distances between the training data and the test object
        dists = np.zeros(len(data_train[0]))
        for i, x in enumerate(data_train[0]):
            dists[i] = distance.distance('Euclidean', x, y)

        # Then we sort the distances
        dists = enumerate(dists)
        ord_dists = sorted(dists, key = lambda dists: dists[1])

        # Finally we get the k-nearest neighbors and select the most repeated class
        knn = ord_dists[:k]
        cls_train = [data_train[1][n[0]] for n in knn]
        cls_num = [(c, cls_train.count(c)) for c in set(cls_train)]
        cls_num_ord = sorted(cls_num, key = lambda cls_num: -cls_num[1])
        cls.append(cls_num_ord[0][0])

    return cls


train = [((2, 3), (2, 4), (6, 1), (7, 2), (4, 3), (5, 1)), (1, 2, 1, 2, 2, 4)]
test = [(2, 1)]
print nearest_neighbor(train, test, 3)