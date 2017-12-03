import utils
import numpy as np
import distance


def nearest_neighbor(data_train, data_test, k, distance_measure, weighted):
    """

    :param distance_measure:
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
            dists[i] = distance.distance(distance_measure, x, y)

        # Instantiate all the weights as one
        weights = [1.0 for i in range(len(dists))]

        # If weights are turned on, calculate the weight of every point
        if weighted:
            for d in range(len(dists)):
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


train = [((2, 3), (2, 4), (6, 1), (7, 2), (4, 3), (5, 1)), (1, 2, 1, 2, 2, 4)]
test = [(2, 1)]
print 'weighted'
print nearest_neighbor(train, test, 3, 'Euclidean', True)
print 'non weighted'
print nearest_neighbor(train, test, 3, 'Euclidean', False)

