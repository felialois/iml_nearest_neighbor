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

    weigths = kNNWeightedAlgorithm.calculate_weights(utils.encode_data(training), training_class, 'Relieff')

    print(weigths)

    cls = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, distance_metric)
    cls2 = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, distance_metric, weigths)

    print distance_metric + ' without weights'
    print(metrics.accuracy(testing_class, cls, 'micro'))
    print(metrics.recall(testing_class, cls))

    print distance_metric + ' with weights'
    print(metrics.accuracy(testing_class, cls2, 'micro'))
    print(metrics.recall(testing_class, cls2))
