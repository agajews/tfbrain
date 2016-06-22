import numpy as np


def labels_to_one_hot(labels, num_categories):
    '''Takes a numpy array of labels and turns it
    into a one-hot numpy array of higher dimension'''
    data = np.zeros(labels.shape + (num_categories,))
    for indices, val in np.ndenumerate(labels.astype('int32')):
        data[indices + (val,)] = 1
    return data
