import sys
import importfile
import kNNAlgorithm

arguments = sys.argv
dir_name = arguments[1]
class_name = arguments[2]
# k = int(arguments[3])
# distance_metric = arguments[4]
# weighted = arguments[5]

# Constants
test_fl = '1'
test_str = 'test'
train_str = 'train'

training, training_class, testing, testing_class = importfile.get_datasets(dir_name, class_name, test_fl, test_str,
                                                                           train_str)

cls = kNNAlgorithm.nearest_neighbor((training, training_class), testing, 5, 'Euclidean', False)
cls2 = kNNAlgorithm.nearest_neighbor((training, training_class), testing, 5, 'Euclidean', True)

print ' Euclidean without weights'
corrects = 0.0
incorrects = 0.0
for res in zip(cls, testing_class):
    if res[0] == res[1]:
        corrects += 1.0
    else:
        incorrects += 1.0
print 'Correct : ' + str(corrects) + '  ' + str(corrects / len(cls)) + '%'
print 'Incorrect : ' + str(incorrects) + '  ' + str(incorrects / len(cls)) + '%'

print ' Euclidean with weights'
corrects = 0.0
incorrects = 0.0
for res in zip(cls2, testing_class):
    if res[0] == res[1]:
        corrects += 1.0
    else:
        incorrects += 1.0
print 'Correct : ' + str(corrects) + '  ' + str(corrects / len(cls2)) + '%'
print 'Incorrect : ' + str(incorrects) + '  ' + str(incorrects / len(cls2)) + '%'



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
