import importfile
import utils
import kNNAlgorithm
import numpy as np
import kNNWeightedAlgorithm
import kNNSelectedAlgorithm
import metrics
import time

# Constants
dir_name = ["datasetsCBR/credit-a", "datasetsCBR/grid", "datasetsCBR/vowel"]
class_name = ["class", "class", "Class"]
test_str = 'test'
train_str = 'train'
k_neighbors = [1, 3, 5, 7]
dist_metrics = ['Hamming', 'Euclidean', 'Cosine', 'Canberra']
policies = ['voting', 'similar']

f = open('results.txt', 'w')
f.write('dataset,algorithm,k,distance metric,policy,efficiency,accuracy,recall')
number_of_folds = 10

for k in k_neighbors:  # Different k neighbors
    print('K :', str(k))
    for dist in dist_metrics:  # Metrics for the distances
        for policy in policies:  # Policies
            for dataset in range(len(dir_name)):  # Datasets
                efficiency_folds = np.zeros((5, number_of_folds))
                accuracy_folds = np.zeros((5, number_of_folds))
                recall_folds = np.zeros((5, number_of_folds))
                for test_fl in range(0, number_of_folds):  # Different folds
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
                    accuracy_folds[0, test_fl] = metrics.accuracy(testing_class, predicted)
                    recall_folds[0, test_fl] = metrics.recall(testing_class, predicted)

                    # Weighted KNN
                    start = time.time()
                    weights_tree = kNNWeightedAlgorithm.calculate_weights(utils.encode_data(training), training_class,
                                                                          'TreeClassifier')
                    predicted = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, dist, policy,
                                                        weights_tree)
                    efficiency_folds[1, test_fl] = time.time() - start
                    accuracy_folds[1, test_fl] = metrics.accuracy(testing_class, predicted)
                    recall_folds[1, test_fl] = metrics.recall(testing_class, predicted)

                    start = time.time()
                    weights_relieff = kNNWeightedAlgorithm.calculate_weights(utils.encode_data(training),
                                                                             training_class, 'Relieff')
                    predicted = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, dist, policy,
                                                              weights_relieff)
                    efficiency_folds[2, test_fl] = time.time() - start
                    accuracy_folds[2, test_fl] = metrics.accuracy(testing_class, predicted)
                    recall_folds[2, test_fl] = metrics.recall(testing_class, predicted)

                    # Selected KNN
                    start = time.time()
                    _, removed_cols = kNNSelectedAlgorithm.remove_features(utils.encode_data(training),
                                                                                   training_class, 'variance')
                    new_test = testing
                    new_train = training
                    for col in reversed(removed_cols):
                        new_test = np.delete(new_test, col, 1)
                        new_train = np.delete(new_train, col, 1)
                    predicted = kNNAlgorithm.nearest_neighbor((new_train, training_class), new_test, k, dist, policy)
                    efficiency_folds[3, test_fl] = time.time() - start
                    accuracy_folds[3, test_fl] = metrics.accuracy(testing_class, predicted)
                    recall_folds[3, test_fl] = metrics.recall(testing_class, predicted)

                    start = time.time()
                    _, removed_cols = kNNSelectedAlgorithm.remove_features(utils.encode_data(training),
                                                                                   training_class, 'L1')
                    new_test = testing
                    new_train = training
                    for col in reversed(removed_cols):
                        new_test = np.delete(new_test, col, 1)
                        new_train = np.delete(new_train, col, 1)
                    predicted = kNNAlgorithm.nearest_neighbor((new_train, training_class), new_test, k, dist, policy)
                    efficiency_folds[4, test_fl] = time.time() - start
                    accuracy_folds[4, test_fl] = metrics.accuracy(testing_class, predicted)
                    recall_folds[4, test_fl] = metrics.recall(testing_class, predicted)

                # Basic KNN
                f.write(dir_name[dataset] + ',Basic KNN,' + str(k) + ',' + dist + ',' + policy + ',')
                efficiency = np.mean(efficiency_folds[0])
                accuracy = np.mean(accuracy_folds[0])
                recall = np.mean(recall_folds[0])
                f.write(str(efficiency) + ',' + str(accuracy) + ',' + str(recall) + '\n')

                # Weighted KNN
                f.write(dir_name[dataset] + ',Weighted KNN TreeClassifier,' + str(k) + ',' + dist + ',' + policy + ',')
                efficiency = np.mean(efficiency_folds[1])
                accuracy = np.mean(accuracy_folds[1])
                recall = np.mean(recall_folds[1])
                f.write(str(efficiency) + ',' + str(accuracy) + ',' + str(recall) + '\n')

                f.write(dir_name[dataset] + ',Weighted KNN Relieff,' + str(k) + ',' + dist + ',' + policy + ',')
                efficiency = np.mean(efficiency_folds[2])
                accuracy = np.mean(accuracy_folds[2])
                recall = np.mean(recall_folds[2])
                f.write(str(efficiency) + ',' + str(accuracy) + ',' + str(recall) + '\n')

                # Selected KNN
                f.write(dir_name[dataset] + ',Selected KNN variance,' + str(k) + ',' + dist + ',' + policy + ',')
                efficiency = np.mean(efficiency_folds[3])
                accuracy = np.mean(accuracy_folds[03])
                recall = np.mean(recall_folds[3])
                f.write(str(efficiency) + ',' + str(accuracy) + ',' + str(recall) + '\n')

                f.write(dir_name[dataset] + ',Selected KNN L1,' + str(k) + ',' + dist + ',' + policy + ',')
                efficiency = np.mean(efficiency_folds[4])
                accuracy = np.mean(accuracy_folds[4])
                recall = np.mean(recall_folds[4])
                f.write(str(efficiency) + ',' + str(accuracy) + ',' + str(recall) + '\n')
