from scipy.io import arff
import numpy as np
import math


def read_file(fl, class_name):
    data, meta = arff.loadarff(fl)

    columns = len(data[0])

    if meta[class_name]:
        columns = columns - 1

    data2 = np.zeros(shape=(len(data), columns))
    data_classes = []

    for i in range(len(meta.names())):
        if meta.names()[i] == class_name:
            for j in range(len(data)):
                data_classes.append(data[j][i])
        else:
            if meta.types()[i] == 'numeric':
                for j in range(len(data)):
                    if math.isnan(data[j][i]):
                        data2[j][i] = -1
                    else:
                        data2[j][i] = data[j][i]
            else:
                values = meta[meta.names()[i]][1]
                for j in range(len(data)):
                    if data[j][i] == '?':
                        data2[j][i] = -1
                    else:
                        data2[j][i] = values.index(data[j][i])

    return data2, data_classes
    # print data2
