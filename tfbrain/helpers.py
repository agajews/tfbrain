import numpy as np

import shelve

import random

import tensorflow as tf

from memory_profiler import profile


def get_input_name(net):
    if len(net.incoming) == 0:
        return net.get_name()
    else:
        assert len(net.incoming) == 1  # no branching
        return get_input_name(net.incoming[0])


def get_output(net, passthrough_vars=None, **kwargs):
    if len(net.incoming) == 0:  # InputLayer
        print(net)
        print(net.get_name())
        if passthrough_vars is None:
            return net.get_output(None, **kwargs)
        else:
            return net.get_output(passthrough_vars[net.get_name()],
                                  **kwargs)
    return net.get_output(*list(map(
        lambda n: get_output(n, passthrough_vars, **kwargs),
        net.incoming)), **kwargs)


def clone(net, passthrough_vars=None, **kwargs):
    print(net.incoming)
    print('passthrough vars')
    print(passthrough_vars)
    if len(net.incoming) == 0:  # InputLayer
        if passthrough_vars is None:
            return net.__class__(**net.config, **kwargs)
        else:
            print('input passthrough')
            print(passthrough_vars[net.get_name()])
            return net.__class__(**net.config, **kwargs,
                                 passthrough=passthrough_vars[net.get_name()])
    if len(net.incoming) == 1:
        incoming = clone(net.incoming[0], passthrough_vars, **kwargs)
        print(incoming)
    else:
        incoming = map(
            lambda i: clone(i, passthrough_vars, **kwargs), net.incoming)
    return net.__class__(incoming,
                         **net.config, **kwargs)


def clone_keep_params(net, passthrough_vars=None, **kwargs):
    print(net.incoming)
    if len(net.incoming) == 0:  # InputLayer
        if passthrough_vars is None:
            return net.__class__(**net.config, **net.params, **kwargs)
        else:
            return net.__class__(**net.config, **net.params, **kwargs,
                                 passthrough=passthrough_vars[net.get_name()])
    if len(net.incoming) == 1:
        incoming = clone_keep_params(
            net.incoming[0], passthrough_vars, **kwargs)
        print(incoming)
    else:
        incoming = map(lambda i: clone_keep_params(i, passthrough_vars,
                                                   **kwargs),
                       net.incoming)
    return net.__class__(incoming,
                         **net.config, **net.params, **kwargs)


def _flatten_layers(net, layers):
    layers += net.incoming
    for layer in net.incoming:
        layers = _flatten_layers(layer, layers)
    return layers


def flatten_layers(net):
    return _flatten_layers(net, [])


def get_input_vars(net):
    input_vars = {}
    for layer in flatten_layers(net):
        print(layer)
        if len(layer.incoming) == 0:  # InputLayer
            print('Input layer!')
            input_vars[layer.get_name()] = layer.placeholder
    return input_vars


def get_output_hidden_vars(net, **kwargs):
    output_hidden_vars = {}
    for layer in flatten_layers(net):
        layer_hidden_var = layer.get_output_hidden_var(**kwargs)
        if layer_hidden_var is not None:
            output_hidden_vars[layer.get_name()] = layer_hidden_var
    return output_hidden_vars


def get_input_hidden_vars(net, **kwargs):
    input_hidden_vars = {}
    for layer in flatten_layers(net):
        layer_hidden_var = layer.get_input_hidden_var(**kwargs)
        if layer_hidden_var is not None:
            input_hidden_vars[layer.get_name()] = layer_hidden_var
    return input_hidden_vars


def get_init_hidden(net, **kwargs):
    input_hidden_vars = {}
    for layer in flatten_layers(net):
        layer_hidden_var = layer.get_init_hidden(**kwargs)
        if layer_hidden_var is not None:
            input_hidden_vars[layer.get_name()] = layer_hidden_var
    return input_hidden_vars


def get_assign_hidden_ops(net, **kwargs):
    assign_hidden_ops = {}
    for layer in flatten_layers(net):
        layer_assigns = layer.get_assign_hidden_op(**kwargs)
        if layer_assigns is not None:
            assign_hidden_ops[layer.get_name()] = layer_assigns
    return assign_hidden_ops


def resolve_sess(sess):
    if sess is None:
        sess = tf.InteractiveSession()

    return sess


def flatten_params(params):
    all_params = []
    for layer_name, layer_params in params.items():
        for param_name, param in layer_params.items():
            # print(param)
            all_params.append(param)
    return all_params


def get_all_params(net):
    all_params = {net.name: net.get_params()}
    if len(net.incoming) > 0:
        for incoming in net.incoming:
            all_params.update(get_all_params(incoming))
    return all_params


def get_trainable_params(net):
    trainable_params = {net.name: net.get_trainable_params()}
    if len(net.incoming) > 0:
        for incoming in net.incoming:
            trainable_params.update(get_all_params(incoming))
    return trainable_params


def get_all_params_copies(all_params):
    copies = {}
    for layer_name, layer_params in all_params.items():
        copies[layer_name] = {}
        for param_name, param in layer_params.items():
            copies[layer_name][param_name] = tf.Variable(
                param.initialized_value())
    return copies


def get_all_params_values(all_params, sess=None):
    sess = resolve_sess(sess)
    all_params_values = {}
    for layer_name, layer_params in all_params.items():
        layer_params_values = {}
        for param_name, param in layer_params.items():
            layer_params_values[param_name] = sess.run(param)
        all_params_values[layer_name] = layer_params_values

    return all_params_values


def get_all_net_params_values(net, sess=None):
    return get_all_params_values(get_all_params(net),
                                 sess=sess)


@profile
def set_all_params_values(net, params_values, sess=None):
    sess = resolve_sess(sess)
    all_params = get_all_params(net)
    for layer_name in all_params:
        for param_name in all_params[layer_name]:
            param = all_params[layer_name][param_name]
            target_param = params_values[layer_name][param_name]
            assign_op = param.assign(target_param)
            sess.run(assign_op)


def set_all_params_ops(dest_params, src_params):
    ops = []
    for layer_name in dest_params:
        for param_name in dest_params[layer_name]:
            dest_param = dest_params[layer_name][param_name]
            src_param = src_params[layer_name][param_name]
            assign_op = dest_param.assign(src_param)
            ops.append(assign_op)
    return ops


def set_all_net_params_ops(net, src_params, sess=None):
    sess = resolve_sess(sess)
    all_params = get_all_params(net)
    ops = []
    for layer_name in all_params:
        for param_name in all_params[layer_name]:
            param = all_params[layer_name][param_name]
            target_param = src_params[layer_name][param_name]
            assign_op = param.assign(target_param)
            ops.append(assign_op)
    return ops


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
    first = inputs[list(inputs.keys())[0]]
    if isinstance(first, np.ndarray):
        length = first.shape[0]
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
    elif isinstance(first, list):
        length = len(first)
        for name in inputs:
            assert len(inputs[name]) == length
        if shuffle:
            indices = list(range(length))
            random.shuffle(indices)

        for start_index in range(0, length - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_index:start_index + batch_size]
            else:
                excerpt = slice(start_index, start_index + batch_size)

            batch = {}
            for name in inputs:
                batch[name] = []
                for index in excerpt:
                    batch[name].append(inputs[name][index])

            yield batch
    else:
        raise Exception('Invalid inputs')


def create_x_feed_dict(input_vars, batch):
    feed_dict = {}
    for name in input_vars.keys():
        feed_dict[input_vars[name]] = batch[name]

    return feed_dict


def create_y_feed_dict(y_var, y_val):
    feed_dict = {}
    feed_dict[y_var] = y_val
    return feed_dict


def create_supp_train_feed_dict(model):
    supp_feed_dict = get_supp_train_feed_dict(model.net)
    return supp_feed_dict


def create_supp_test_feed_dict(model):
    supp_feed_dict = get_supp_test_feed_dict(model.net)
    return supp_feed_dict


def create_minibatch_iterator(train_xs,
                              train_y,
                              batch_preprocessor,
                              batch_size,
                              train_mask=None):
    inputs = {}
    inputs.update(train_xs)
    if train_y is not None:
        inputs['y'] = train_y
    if train_mask is not None:
        inputs['mask'] = train_mask
    minibatches = iterate_minibatches(
        inputs, batch_size=batch_size)

    if batch_preprocessor is None:
        return minibatches
    else:
        return map(batch_preprocessor, minibatches)


def softmax(np_array):
    maxs = np.amax(np_array, axis=1)
    maxs = maxs.reshape(maxs.shape[0], 1)
    e = np.exp(np_array - maxs)
    dist = e / np.sum(e, axis=1)
    return dist


def avg_over_batches(minibatches, fn):
        res = 0
        num_batches = 0
        for batch in minibatches:
            res += fn(batch)
            num_batches += 1
        return res / num_batches


def rand_dict_sample(dictionary, num_samples, keys=None):
    if keys is None:
        keys = list(dictionary.keys())
    indices = random.sample(range(len(keys)), num_samples)
    sample = []
    for index in indices:
        sample.append(dictionary[keys[index]])
    return sample


class NumpyDeque(object):
    def __init__(self, item_shape, maxlen, **kwargs):
        self.array = np.zeros((maxlen,) + item_shape, **kwargs)
        self.maxlen = maxlen
        self.num_items = 0
        self.frontier = 0

    def append(self, item):
        if self.num_items < self.maxlen:
            self.array[self.num_items] = item
            self.num_items += 1
        else:
            self.array[self.frontier] = item
            self.frontier = (self.frontier + 1) % self.maxlen

    def get(self, pos):
        return self.array[pos]

    def __getitem__(self, arg):
        return self.array[arg]

    def __len__(self):
        return self.num_items

    def sample(self, sample_size):
        keys = random.sample(range(self.num_items), sample_size)
        return self.array[keys]


class PersistentDeque(object):
    def __init__(self, db_fnm, maxlen, cache_size, update_freq=100):
        self.maxlen = maxlen
        self.shelf = shelve.open(
            db_fnm, writeback=True)
        self.num_items = 0
        self.frontier = 0
        self.update_num = 0
        self.update_freq = update_freq
        self.keys = []

    def append(self, item):
        if self.num_items < self.maxlen:
            self.shelf[str(self.num_items)] = item
            self.keys.append(str(self.num_items))
            self.num_items += 1
        else:
            self.shelf[str(self.frontier)] = item
            self.frontier = (self.frontier + 1) % self.maxlen
        if self.update_num % self.update_freq == 0:
            self.shelf.sync()
        self.update_num += 1

    def get(self, pos):
        return self.shelf[str(pos)]

    def __getitem__(self, arg):
        return self.get(arg)

    def __len__(self):
        return self.num_items

    def sample(self, num_samples):
        return rand_dict_sample(self.shelf, num_samples, keys=self.keys)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
