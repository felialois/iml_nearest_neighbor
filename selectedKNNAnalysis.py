import importfile
import utils
import numpy as np
import kNNSelectedAlgorithm

# Parameters
dir_name = ["datasetsCBR/credit-a", "datasetsCBR/grid", "datasetsCBR/vowel"]
class_name = ["class", "class", "Class"]
test_str = 'test'
train_str = 'train'
number_of_folds = 10
# Best parameters found
k = 1
dist = 'Euclidean'
policy = 'voting'

f_selknn = open('selectedKNN_delfeatures.txt', 'w')
f_selknn.write('dataset,Selected KNN variance avg. removed features(Total),Selected KNN L1 avg. removed features(Total)\n')

for dataset in range(len(dir_name)):  # Datasets
    print('Dataset ' + dir_name[dataset])
    f_selknn.write(dir_name[dataset] + ',')

    variance_del = np.zeros(number_of_folds)
    l1_del = np.zeros(number_of_folds)

    for test_fl in range(number_of_folds):  # Different folds

        training, training_class, _, testing_class = importfile.get_datasets(dir_name[dataset],
                                                                                   class_name[dataset],
                                                                                   str(test_fl),
                                                                                   test_str, train_str)

        training = np.transpose(utils.normalize_columns(np.transpose(training)))

        # Selected KNN variance
        kept_cols, removed_cols = kNNSelectedAlgorithm.remove_features(utils.encode_data(training),
                                                               training_class, 'variance')
        variance_del[test_fl] = len(removed_cols)

        # Selected KNN L1
        kept_cols, removed_cols = kNNSelectedAlgorithm.remove_features(utils.encode_data(training),
                                                               training_class, 'L1')
        l1_del[test_fl] = len(removed_cols)

    f_selknn.write(str(np.average(variance_del)) + '(' + str(len(removed_cols) + len(kept_cols)) + '),')
    f_selknn.write(str(np.average(l1_del)) + '(' + str(len(removed_cols) + len(kept_cols)) + ')\n')

f_selknn.close()