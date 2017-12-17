import importfile
import utils
import kNNAlgorithm
import numpy as np
import kNNWeightedAlgorithm
import kNNSelectedAlgorithm
import metrics
import time

# Parameters
dir_name = ["datasetsCBR/credit-a", "datasetsCBR/grid", "datasetsCBR/vowel"]
class_name = ["class", "class", "Class"]
test_str = 'test'
train_str = 'train'
# Best parameters found
k = 1
dist = 'Euclidean'
policy = 'voting'

# Files with the results
f_eff = open('paired_ttest_eff.txt', 'w')
f_eff.write('dataset,fold,basic knn(baseline),weighted knn treeCl(diff),weighted knn relieff(diff),'
        'selected knn variance(diff),selected knn L1(diff)\n')
f_acc = open('paired_ttest_acc.txt', 'w')
f_acc.write('dataset,fold,basic knn(baseline),weighted knn treeCl(diff),weighted knn relieff(diff),'
        'selected knn variance(diff),selected knn L1(diff)\n')
f_rec = open('paired_ttest_rec.txt', 'w')
f_rec.write('dataset,fold,basic knn(baseline),weighted knn treeCl(diff),weighted knn relieff(diff),'
        'selected knn variance(diff),selected knn L1(diff)\n')
f_rank = open('paired_ttest_rank.txt', 'w')
f_rank.write('dataset,fold,basic knn,weighted knn treeCl,weighted knn relieff,selected knn variance,selected knn L1\n')

for dataset in range(len(dir_name)):  # Datasets
    print('Dataset ' + dir_name[dataset])
    for test_fl in range(0, 10):  # Different folds
        efficiencies = np.zeros(5)
        accuracies = np.zeros(5)
        recalls = np.zeros(5)

        print("Fold number " + str(test_fl))

        training, training_class, testing, testing_class = importfile.get_datasets(dir_name[dataset],
                                                                                   class_name[dataset],
                                                                                   str(test_fl),
                                                                                   test_str, train_str)

        training = np.transpose(utils.normalize_columns(np.transpose(training)))
        testing = np.transpose(utils.normalize_columns(np.transpose(testing)))

        # Basic KNN (baseline)
        start = time.time()
        predicted = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, dist, policy)
        efficiencies[0] = time.time() - start
        accuracies[0] = metrics.accuracy(testing_class, predicted)
        recalls[0] = metrics.recall(testing_class, predicted)

        # Weighted KNN
        start = time.time()
        weights_tree = kNNWeightedAlgorithm.calculate_weights(utils.encode_data(training), training_class,
                                                              'TreeClassifier')
        predicted = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, dist, policy,
                                            weights_tree)
        efficiencies[1] = time.time() - start
        accuracies[1] = metrics.accuracy(testing_class, predicted)
        recalls[1] = metrics.recall(testing_class, predicted)

        start = time.time()
        weights_relieff = kNNWeightedAlgorithm.calculate_weights(utils.encode_data(training),
                                                                 training_class, 'Relieff')
        predicted = kNNAlgorithm.nearest_neighbor((training, training_class), testing, k, dist, policy,
                                                  weights_relieff)
        efficiencies[2] = time.time() - start
        accuracies[2] = metrics.accuracy(testing_class, predicted)
        recalls[2] = metrics.recall(testing_class, predicted)

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
        efficiencies[3] = time.time() - start
        accuracies[3] = metrics.accuracy(testing_class, predicted)
        recalls[3] = metrics.recall(testing_class, predicted)

        start = time.time()
        _, removed_cols = kNNSelectedAlgorithm.remove_features(utils.encode_data(training),
                                                                       training_class, 'L1')
        new_test = testing
        new_train = training
        for col in reversed(removed_cols):
            new_test = np.delete(new_test, col, 1)
            new_train = np.delete(new_train, col, 1)
        predicted = kNNAlgorithm.nearest_neighbor((new_train, training_class), new_test, k, dist, policy)
        efficiencies[4] = time.time() - start
        accuracies[4] = metrics.accuracy(testing_class, predicted)
        recalls[4] = metrics.recall(testing_class, predicted)

        # Here we write the files, applying paired t-test with the basic knn algorithm as baseline
        # Between parenthesis the difference with the baseline
        # Finally we get the rank based on the accuracy

        # Efficiencies
        f_eff.write(dir_name[dataset] + ',' + str(test_fl) + ',' + str(efficiencies[0]) + '(0.0),')
        f_eff.write(str(efficiencies[1]) + '(' + str(efficiencies[1] - efficiencies[0]) + '),')
        f_eff.write(str(efficiencies[2]) + '(' + str(efficiencies[2] - efficiencies[0]) + '),')
        f_eff.write(str(efficiencies[3]) + '(' + str(efficiencies[3] - efficiencies[0]) + '),')
        f_eff.write(str(efficiencies[4]) + '(' + str(efficiencies[4] - efficiencies[0]) + '),\n')

        # Accuracies
        diff_acc = [0.0, accuracies[1] - accuracies[0], accuracies[2] - accuracies[0], accuracies[3] - accuracies[0],
                    accuracies[4] - accuracies[0]]
        f_acc.write(dir_name[dataset] + ',' + str(test_fl) + ',' + str(accuracies[0]) + '(0.0),')
        f_acc.write(str(accuracies[1]) + '(' + str(diff_acc[1]) + '),')
        f_acc.write(str(accuracies[2]) + '(' + str(diff_acc[2]) + '),')
        f_acc.write(str(accuracies[3]) + '(' + str(diff_acc[3]) + '),')
        f_acc.write(str(accuracies[4]) + '(' + str(diff_acc[4]) + '),\n')

        # Recalls
        f_rec.write(dir_name[dataset] + ',' + str(test_fl) + ',' + str(recalls[0]) + '(0.0),')
        f_rec.write(str(recalls[1]) + '(' + str(recalls[1] - recalls[0]) + '),')
        f_rec.write(str(recalls[2]) + '(' + str(recalls[2] - recalls[0]) + '),')
        f_rec.write(str(recalls[3]) + '(' + str(recalls[3] - recalls[0]) + '),')
        f_rec.write(str(recalls[4]) + '(' + str(recalls[4] - recalls[0]) + '),\n')

        # Ranks
        algorithms = ['basic knn', 'weighted knn treeCl', 'weighted knn relieff', 'selected knn variance', 'selected knn L1']
        diff_acc = enumerate(diff_acc)
        diff_acc_rank = sorted(diff_acc, key=lambda diff_acc: -diff_acc[1])
        # We calculate the ranks
        ranks = [0] * 5
        for i in range(len(diff_acc_rank)):
            if i > 0:
                # If both algorithms have the same difference they have the same rank
                if diff_acc_rank[i][1] == diff_acc_rank[i-1][1]:
                    ranks[diff_acc_rank[i][0]] = ranks[diff_acc_rank[i-1][0]]
                else:
                    ranks[diff_acc_rank[i][0]] = i + 1
            else:
                ranks[diff_acc_rank[i][0]] = i+1
        # Now we normalize the ranks
        count_ranks = {rank: ranks.count(rank) for rank in set(ranks)}
        for i in range(len(ranks)):
            # If there are more than 1 algorithm with the same rank we normalize it
            if count_ranks[ranks[i]] > 1:
                ranks[i] += 1.0/count_ranks[ranks[i]]

        # Now we print all
        f_rank.write(dir_name[dataset] + ',' + str(test_fl))
        for rank in ranks:
            f_rank.write(',' + str(rank))
        f_rank.write('\n')

f_eff.close()
f_acc.close()
f_rec.close()
f_rank.close()