import numpy as np


def get_output(layer):
    if len(layer.incoming) == 0:
        return layer.get_output(None)
    else:
        return layer.get_output(*map(get_output,
                                     layer.incoming))


def get_supp_train_feed_dict(layer):
    if len(layer.incoming) == 0:
        return layer.get_supp_train_feed_dict()
    else:
        feed_dict = layer.get_supp_train_feed_dict()
        for incoming in layer.incoming:
            feed_dict.update(get_supp_train_feed_dict(incoming))
        return feed_dict


def get_supp_test_feed_dict(layer):
    if len(layer.incoming) == 0:
        return layer.get_supp_test_feed_dict()
    else:
        feed_dict = layer.get_supp_test_feed_dict()
        for incoming in layer.incoming:
            feed_dict.update(get_supp_test_feed_dict(incoming))
        return feed_dict


def iterate_minibatches(inputs, batch_size=128, shuffle=True):
    length = inputs[list(inputs.keys())[0]].shape[0]
    for name in inputs:
        assert inputs[name].shape[0] == length

    if shuffle:
        indices = np.arange(length)
        np.random.shuffle(indices)

    for start_index in range(0, length - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_index:start_index + batch_size]
        else:
            excerpt = slice(start_index, start_index + batch_size)

        batch = {}
        for name in inputs:
            batch[name] = inputs[name][excerpt]

        yield batch
