import numpy as np
import distance


def nearest_neighbor(data_train, data_test, k, distance_measure, policy, weights=[]):
    """
    Function of the k nearest neighbor classification
    :param distance_measure:
    :param data_train: (data, class)
    :param data_test:  (data)
    :param k:
    :param weights:

    :return: prediction given for the test data
    """
    if len(weights) == 0:
        weights = [1.0 for i in range(len(data_train[0]))]
    cls = []
    for y in data_test:
        # We calculate all the distances between the training data and the test object
        dists = np.zeros(len(data_train[0]))
        for i, x in enumerate(data_train[0]):
            dists[i] = distance.distance(distance_measure, x, y, weights)

        # Apply the weights to the distances
        dists = [dists[y] for y in range(len(dists))]
        # Then we sort the distances
        dists = enumerate(dists)
        ord_dists = sorted(dists, key=lambda dists: dists[1])

        # Finally we get the k-nearest neighbors and select the predicted class
        knn = ord_dists[:k]
        cls_train = [data_train[1][n[0]] for n in knn]
        cls_num = [(c, cls_train.count(c)) for c in set(cls_train)]
        if policy == 'voting':
            cls_num_ord = sorted(cls_num, key=lambda cls_num: -cls_num[1])
            cls.append(cls_num_ord[0][0])
        elif policy == 'similar':
            cls_dist = {c:0 for c in set(cls_train)}
            # Sum of the distances per class
            for neighbor in knn:
                cls_dist[data_train[1][neighbor[0]]] += neighbor[1]
            # Mean of the distances per class
            for cls_n in cls_num:
                cls_dist[cls_n[0]] /= cls_n[1]
            cls_dist_tuples = [x for x in cls_dist.items()]
            cls_dist_ord = sorted(cls_dist_tuples, key=lambda cls_dist_tuples: -cls_dist_tuples[1])
            cls.append(cls_dist_ord[0][0])
        else:
            raise ValueError

    return cls
