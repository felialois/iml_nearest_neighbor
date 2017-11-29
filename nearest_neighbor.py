import utils


def nearest_neighbor(data):
    data = zip(*utils.normalize_columns(zip(*data)))
