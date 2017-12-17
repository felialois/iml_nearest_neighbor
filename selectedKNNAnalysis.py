import importfile
import utils
import numpy as np
import kNNSelectedAlgorithm

# Parameters
dir_name = ["datasetsCBR/credit-a", "datasetsCBR/grid", "datasetsCBR/vowel"]
class_name = ["class", "class", "Class"]
test_str = 'test'
train_str = 'train'
fold = 0
# Best parameters found
k = 1
dist = 'Euclidean'
policy = 'voting'

f_selknn = open('selectedKNN_delfeatures.txt', 'w')
f_selknn.write('dataset,Selected KNN variance removed features(Total),Selected KNN L1 removed features(Total)\n')

for dataset in range(len(dir_name)):  # Datasets
    print('Dataset ' + dir_name[dataset])
    f_selknn.write(dir_name[dataset] + ',')

    training, training_class, _, testing_class = importfile.get_datasets(dir_name[dataset],
                                                                               class_name[dataset],
                                                                               str(fold),
                                                                               test_str, train_str)

    training = np.transpose(utils.normalize_columns(np.transpose(training)))

    # Selected KNN variance
    kept_cols, removed_cols = kNNSelectedAlgorithm.remove_features(utils.encode_data(training),
                                                           training_class, 'variance')
    f_selknn.write(str(len(removed_cols)) + '(' + str(len(removed_cols) + len(kept_cols)) + '),')

    # Selected KNN L1
    kept_cols, removed_cols = kNNSelectedAlgorithm.remove_features(utils.encode_data(training),
                                                           training_class, 'L1')

    f_selknn.write(str(len(removed_cols)) + '(' + str(len(removed_cols) + len(kept_cols)) + ')\n')

f_selknn.close()