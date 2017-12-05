import utils
import numpy as np
import distance


def nearest_neighbor(data_train, data_test, k, distance_measure, weighted):
    """

    :param distance_measure:
    :param data_train: (data, class)
    :param data_test:  (data)
    :param k:
    :param weighted:

    :return:
    """
    cls = []
    for y in data_test:
        # We alculate all the distances between the training data and the test object
        dists = np.zeros(len(data_train[0]))
        for i, x in enumerate(data_train[0]):
            dists[i] = distance.distance(distance_measure, x, y)

        # Instantiate all the weights as one
        weights = [1.0 for i in range(len(dists))]

        # If weights are turned on, calculate the weight of every point
        if weighted:
            for d in range(len(dists)):
                if dists[d] == 0:
                    weights[d] = 1
                else:
                    weights[d] = 1 / (dists[d] ** 2)
            # Use all the training data for the weighted KNN
            k = len(dists)

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
