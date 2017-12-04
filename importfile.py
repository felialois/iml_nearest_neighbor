from scipy.io import arff
import numpy as np
import math
from os import listdir
from os.path import isfile, join


def read_file(fl, class_name):
    """

    :param fl: name of the file
    :param class_name: name of the column class
    :return: (data, data_classes) the data with the nominal values transformed and an array containing the class column
    """
    data, meta = arff.loadarff(fl)
    data = np.asarray(data).tolist()

    columns = len(data[0])

    if meta[class_name]:
        columns = columns - 1

    data_classes = []

    class_col = meta.names().index(class_name)

    data_classes = [data[i][class_col] for i in range(len(data))]
    cols = range(len(data[0]))
    cols.remove(class_col)
    data2 = [data[i][0:class_col] + data[i][class_col + 1:] for i in range(len(data))]
    return data2, data_classes


def get_datasets(directory_name, class_name, test_fl, test_str, train_str):
    """
    Read from the directory, extract all the training files and the test files

    :param directory_name: name of the directory with all the files
    :param class_name: Name of the column with the class attribute
    :param test_fl: number of file to be used for testing
    :param test_str: unique identifier for the testing files
    :param train_str: unique identifier for the training files
    :return: (training, training_class, testing, testing_class) both datasets with their respective classes
    """
    train_files = []
    test_file = ""
    files = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]
    for fl in files:
        if test_fl in fl.lower() and test_str in fl.lower():
            test_file = fl.lower()
        elif train_str in fl.lower():
            train_files.append(fl.lower())

    training_dts = []
    training_class = []
    for train_file in train_files:
        tr, cl = read_file(directory_name + '/' + train_file, class_name)
        training_dts.append(tr)
        training_class += cl

    testing, testing_class = read_file(directory_name + '/' + test_file, class_name)
    training = training_dts[0]
    for i in range(1, len(training_dts)):
        training = np.concatenate((training, training_dts[i]), axis=0)
    return training, training_class, testing, testing_class
