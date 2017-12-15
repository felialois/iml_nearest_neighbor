import sys
import importfile
import utils
import kNNAlgorithm
import numpy as np
import kNNWeightedAlgorithm
import kNNSelectedAlgorithm
import metrics

# arguments = sys.argv
# dir_name = arguments[1]
# class_name = arguments[2]
# k = int(arguments[3])
# distance_metric = arguments[4]
# weighted = arguments[5]
dir_name = "datasetsCBR/grid"
class_name = "class"
k = 3
distance_metric = "Euclidean"

# Constants
test_str = 'test'
train_str = 'train'

for test_fl in range(0, 10):
    print("Fold number " + str(test_fl))
    training, training_class, testing, testing_class = importfile.get_datasets(dir_name, class_name, str(test_fl),
                                                                               test_str,
                                                                               train_str)

    training = np.transpose(utils.normalize_columns(np.transpose(training)))
    testing = np.transpose(utils.normalize_columns(np.transpose(testing)))

    weigths = kNNWeightedAlgorithm.calculate_weights(utils.encode_data(training), training_class, 'TreeClassifier')

    print(weigths)

    cls = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, distance_metric)
    cls2 = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, distance_metric, weigths)

    print distance_metric + ' without weights'
    print(metrics.accuracy(testing_class, cls, 'micro'))
    print(metrics.recall(testing_class, cls))

    print distance_metric + ' with weights'
    print(metrics.accuracy(testing_class, cls2, 'micro'))
    print(metrics.recall(testing_class, cls2))

# dts, cls = read_file(file_name, class_name)
# new_dts = pca(dts, 'data', cls, k, cols)
#
# m = np.mean(dts, axis=0)
# p = PCA(n_components=3)
# x = p.fit_transform(dts)
# plot_data(x, cls, 'transformed_data_sk.png', cols)
#
# Xhat = np.dot(p.transform(dts), p.components_)
# Xhat += m
# plot_data(Xhat, cls, 'retransformed_data_sk.png', cols)
# print '\n\n'
