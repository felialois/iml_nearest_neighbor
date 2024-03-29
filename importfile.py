from scipy.io import arff
import pandas as pd
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
    data = pd.DataFrame(data).as_matrix()

    data_classes = []

    class_col = meta.names().index(class_name)

    data_classes = [data[i][class_col] for i in range(len(data))]
    data2 = np.delete(data, class_col, 1)
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
    train_file = ""
    test_file = ""
    files = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]
    for fl in files:
        if test_fl in fl.lower() and test_str in fl.lower():
            test_file = fl.lower()
        elif train_str in fl.lower():
            train_file = fl.lower()

    training, training_class = read_file(directory_name + '/' + train_file, class_name)
    testing, testing_class = read_file(directory_name + '/' + test_file, class_name)

    return training, training_class, testing, testing_class
