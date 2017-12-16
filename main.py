import sys
import importfile
import utils
import kNNAlgorithm
import numpy as np
import kNNWeightedAlgorithm
import kNNSelectedAlgorithm
import metrics
import time

# arguments = sys.argv
# dir_name = arguments[1]
# class_name = arguments[2]
# k = int(arguments[3])
# distance_metric = arguments[4]
# weighted = arguments[5]

# Constants
dir_name = ["datasetsCBR/grid", "datasetsCBR/grid"]
class_name = ["class", "class"]
test_str = 'test'
train_str = 'train'
k_neighbors = [1,3,5,7]
dist_metrics = ['Hamming', 'Euclidean', 'Cosine', 'Canberra']
policies = ['voting', 'similar']

f = open('results.txt', 'w')
f.write('dataset,k,distance metric,policy,efficiency,accuracy,recall')

for k in k_neighbors:                           # Different k neighbors
    for dist in dist_metrics:                   # Metrics for the distances
        for policy in policies:                 # Policies
            for dataset in range(2):            # Datasets
                efficiency_folds = np.zeros((5, 10))
                accuracy_folds = np.zeros((5, 10))
                recall_folds = np.zeros((5, 10))
                for test_fl in range(0, 10):    # Different folds
                    print("Fold number " + str(test_fl))

                    training, training_class, testing, testing_class = importfile.get_datasets(dir_name[dataset],
                                                                                               class_name[dataset],
                                                                                               str(test_fl),
                                                                                               test_str, train_str)

                    training = np.transpose(utils.normalize_columns(np.transpose(training)))
                    testing = np.transpose(utils.normalize_columns(np.transpose(testing)))

                    # Basic KNN
                    start = time.time()
                    predicted = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, dist, policy)
                    efficiency_folds[0, test_fl] = time.time() - start
                    accuracy_folds[0, test_fl] = metrics.accuracy(training_class, predicted)
                    recall_folds[0, test_fl] = metrics.recall(training_class, predicted)


                    # Weighted KNN
                    start = time.time()
                    weights_tree = kNNWeightedAlgorithm.calculate_weights(utils.encode_data(training), training_class, 'TreeClassifier')
                    res = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, dist, policy, weights_tree)
                    efficiency_folds[1, test_fl] = time.time() - start
                    accuracy_folds[1, test_fl] = metrics.accuracy(training_class, predicted)
                    recall_folds[1, test_fl] = metrics.recall(training_class, predicted)


                    start = time.time()
                    weights_relieff = kNNWeightedAlgorithm.calculate_weights(utils.encode_data(training), training_class, 'Relieff')
                    predicted = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, dist, policy, weights_relieff)
                    efficiency_folds[2, test_fl] = time.time() - start
                    accuracy_folds[2, test_fl] = metrics.accuracy(training_class, predicted)
                    recall_folds[2, test_fl] = metrics.recall(training_class, predicted)


                    # Selected KNN
                    start = time.time()
                    new_train = kNNSelectedAlgorithm.remove_features(training, training_class, 'variance')
                    new_test =
                    predicted = kNNAlgorithm.nearest_neighbor((new_train, training_class), new_test, k, dist, policy)
                    efficiency_folds[3, test_fl] = time.time() - start
                    accuracy_folds[3, test_fl] = metrics.accuracy(training_class, predicted)
                    recall_folds[3, test_fl] = metrics.recall(training_class, predicted)


                    start = time.time()
                    new_train = kNNSelectedAlgorithm.remove_features(training, training_class, 'L1')
                    new_test =
                    predicted = kNNAlgorithm.nearest_neighbor((new_train, training_class), new_test, k, dist, policy)
                    efficiency_folds[4, test_fl] = time.time() - start
                    accuracy_folds[4, test_fl] = metrics.accuracy(training_class, predicted)
                    recall_folds[4, test_fl] = metrics.recall(training_class, predicted)
