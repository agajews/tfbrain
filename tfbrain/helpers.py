import numpy as np

import tensorflow as tf

from memory_profiler import profile


def get_output(net):
    if len(net.incoming) == 0:
        return net.get_output(None)
    else:
        return net.get_output(*list(map(get_output,
                                        net.incoming)))


def resolve_sess(sess):
    if sess is None:
        sess = tf.InteractiveSession()

    return sess


def get_all_params(net):
    all_params = {net.name: net.params}
    if len(net.incoming) > 0:
        for incoming in net.incoming:
            all_params.update(get_all_params(incoming))
    return all_params


def get_all_params_copies(all_params):
    copies = {}
    for layer_name in all_params:
        copies[layer_name] = {}
        for param_name in all_params[layer_name]:
            param = all_params[layer_name][param_name]
            copies[layer_name][param_name] = tf.Variable(param.initialized_value())
    return copies


def get_all_params_values(all_params, sess=None):
    sess = resolve_sess(sess)
    all_params_values = {}
    for layer_name in all_params:
        layer_params = {}
        for param_name in all_params[layer_name]:
            param = all_params[layer_name][param_name]
            layer_params[param_name] = sess.run(param)
        all_params_values[layer_name] = layer_params

    return all_params_values


def get_all_net_params_values(net, sess=None):
    sess = resolve_sess(sess)
    all_params = get_all_params(net)
    all_params_values = {}
    for layer_name in all_params:
        layer_params = {}
        for param_name in all_params[layer_name]:
            param = all_params[layer_name][param_name]
            layer_params[param_name] = sess.run(param)
        all_params_values[layer_name] = layer_params

    return all_params_values


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


def set_all_params_ops(dest_params, src_params, sess=None):
    sess = resolve_sess(sess)
    ops = []
    for layer_name in dest_params:
        for param_name in dest_params[layer_name]:
            param = dest_params[layer_name][param_name]
            target_param = src_params[layer_name][param_name]
            assign_op = param.assign(target_param)
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


def create_x_feed_dict(input_vars, batch):
    feed_dict = {}
    for name in batch:
        if not name == 'y':
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
    inputs['y'] = train_y
    if train_mask is not None:
        inputs['mask'] = train_mask
    minibatches = iterate_minibatches(
        inputs, batch_size=batch_size)

    if batch_preprocessor is None:
        return minibatches
    else:
        return map(batch_preprocessor, minibatches)


def avg_over_batches(minibatches, fn):
        res = 0
        num_batches = 0
        for batch in minibatches:
            res += fn(batch)
            num_batches += 1
        return res / num_batches
