from tensorflow.models.rnn.ptb import reader

import numpy as np


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


def load_data(seqlength=20):
    raw_data = reader.ptb_raw_data('data/ptb')
    train_data, val_data, test_data, _ = raw_data
    train_data = indices_to_seq_data(train_data, seqlength)
    val_data = indices_to_seq_data(val_data, seqlength)
    return {'train': train_data,
            'test': val_data}
