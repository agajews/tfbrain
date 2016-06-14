import numpy as np


def labels_to_one_hot(labels, num_categories):
    '''Takes a numpy array of labels and turns it
    into a one-hot numpy array of higher dimension'''
    data = np.zeros(labels.shape + (num_categories,))
    indices = []
    for i in range(len(labels.shape)):
        indices.append(np.arange(labels.shape[i]))
    indices.append(labels)
    data[indices] = 1
    return data


def indices_to_seq_data(indices, seqlength):
    num_examples = len(indices) - seqlength
    text = np.zeros((num_examples, seqlength))
    targets = np.zeros((num_examples,),
                       dtype='int32')
    for example_num in range(0, len(indices) - seqlength):
        start = example_num
        end = start + seqlength
        text[example_num, :] = indices[start:end]
        targets[example_num] = indices[end]
    return {'text': text,
            'targets': targets}
