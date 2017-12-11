import utils
import numpy as np
import distance


def nearest_neighbor(data_train, data_test, k, distance_measure, weights=[]):
    """

    :param distance_measure:
    :param data_train: (data, class)
    :param data_test:  (data)
    :param k:
    :param weights:

    :return:
    """
    if len(weights) == 0:
        weights = [1.0 for i in range(len(data_train[0]))]
    cls = []
    for y in data_test:
        # We alculate all the distances between the training data and the test object
        dists = np.zeros(len(data_train[0]))
        for i, x in enumerate(data_train[0]):
            dists[i] = distance.distance(distance_measure, x, y)

        # Apply the weights to the distances
        dists = [dists[y] for y in range(len(dists))]
        # Then we sort the distances
        dists = enumerate(dists)
        ord_dists = sorted(dists, key=lambda distrs: distrs[1])

        # Finally we get the k-nearest neighbors and select the most repeated class
        knn = ord_dists[:k]
        cls_train = [data_train[1][n[0]] for n in knn]

        cls_num = [(c, apply_weights(c, cls_train, weights)) for c in set(cls_train)]
        cls_num_ord = sorted(cls_num, key=lambda cols_num: -cols_num[1])
        cls.append(cls_num_ord[0][0])

    return cls


def apply_weights(label, class_labels, weights):
    result = 0.0
    for cl_lb in range(len(class_labels)):
        if class_labels[cl_lb] == label:
            result += weights[cl_lb]
    return result

# train = [((2, 3, 'A'), (2, 4, 'B'), (6, 1, 'A'), (7, 2, 'C'), (4, 3, 'D'), (5, 1, 'E')), (1, 2, 1, 2, 2, 4)]
# test = [(2, 1, 'A')]
# print 'weighted'
# print nearest_neighbor(train, test, 3, 'Euclidean', True)
# print 'non weighted'
# print nearest_neighbor(train, test, 3, 'Euclidean', False)
