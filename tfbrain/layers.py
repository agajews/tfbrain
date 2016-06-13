import tensorflow as tf

from functools import reduce

from tfbrain import nonlin, init


class Layer(object):

    def get_output(self, incoming_var):
        '''This method takes a TF variable
        (presumably the output from the current
        layer's incoming layer, but not necessarily)
        and uses it to get the output of the current
        layer)'''
        raise NotImplementedError()

    def get_supp_train_feed_dict(self):
        '''Returns any supplementary feed_dict
        items (for filling TF placeholders)
        (i.e. a dropout layer might set keep_prob)'''
        return {}

    def get_supp_test_feed_dict(self):
        '''Returns any supplementary feed_dict
        items (for filling TF placeholders)
        (i.e. a dropout layer might set keep_prob)'''
        return {}

    def resolve_param(self, param, expected_shape, init):
        if param is None:
            return tf.Variable(init(expected_shape))
        else:
            assert tuple(param.get_shape().as_list()) == expected_shape
            return param


class InputLayer(Layer):

    def __init__(self, shape=None):
        self.output_shape = shape
        self.initialize_placeholder()
        self.incoming = []

    def initialize_placeholder(self):
        self.placeholder = tf.placeholder(tf.float32,
                                          shape=self.output_shape)

    def get_output(self, incoming_var):
        return self.placeholder


class FullyConnectedLayer(Layer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin=nonlin.relu,
                 W_init=init.truncated_normal,
                 b_init=init.constant,
                 W=None,
                 b=None):
        self.check_compatible(incoming)
        self.incoming = [incoming]
        self.incoming_shape = incoming.output_shape

        self.num_nodes = num_nodes
        self.nonlin = nonlin

        self.initialize_params(W, b, W_init, b_init)

        self.output_shape = (None, num_nodes)

    def check_compatible(self, incoming):
        if not len(incoming.output_shape) == 2:
            raise Exception('Incoming layer\'s output shape %s \
                            incompatible, try passing it through \
                            a FlattenLayer first'
                            % str(incoming.output_shape))

    def initialize_params(self, W, b, W_init, b_init):
        W_shape = (self.incoming_shape[1], self.num_nodes)
        b_shape = (self.num_nodes,)

        self.W = self.resolve_param(W, W_shape, W_init)
        self.b = self.resolve_param(b, b_shape, b_init)

    def get_output(self, incoming_var):
        return self.nonlin(
            tf.matmul(incoming_var, self.W) + self.b)


class FlattenLayer(Layer):

    def __init__(self, incoming):
        self.incoming = [incoming]
        self.output_shape = self.flatten_shape(incoming.output_shape)
        print('flatten_output_shape: %s' % str(self.output_shape))

    def flatten_shape(self, shape):
        return (None,
                reduce(lambda x, y: x * y,
                       shape[1:]))

    def get_output(self, incoming_var):
        return tf.reshape(incoming_var, (-1,) + self.output_shape[1:])


class ReshapeLayer(Layer):

    def __init__(self, incoming, shape):
        self.incoming = [incoming]
        self.output_shape = list(map(lambda d: d if d is not None else -1,
                                     shape))

    def get_output(self, incoming_var):
        return tf.reshape(incoming_var, self.output_shape)


def conv_output_length(input_length,
                       filter_length,
                       stride,
                       pad):
    if pad == 'VALID':
        output_length = input_length - filter_length + 1
    elif pad == 'SAME':
        output_length = input_length
    else:
        raise Exception('Invalid padding algorithm %s' % pad)

    return (output_length + stride - 1) // stride


class Conv2DLayer(Layer):

    def __init__(self,
                 incoming,
                 filter_size,
                 num_filters,
                 inner_strides=(1, 1),
                 pad='SAME',
                 nonlin=nonlin.relu,
                 W_init=init.truncated_normal,
                 b_init=init.constant,
                 W=None,
                 b=None):

        self.incoming = [incoming]
        self.nonlin = nonlin
        num_channels = incoming.output_shape[3]
        self.output_shape = self.calc_output_shape(incoming,
                                                   filter_size,
                                                   num_filters,
                                                   num_channels,
                                                   inner_strides,
                                                   pad)

        self.strides = (1,) + inner_strides + (1,)
        self.pad = pad

        W_shape = filter_size + (num_channels, num_filters)
        print('W_conv_shape: %s' % str(W_shape))
        b_shape = (num_filters,)
        print('b_conv_shape: %s' % str(b_shape))
        print('conv_output_shape: %s' % str(self.output_shape))

        self.W = self.resolve_param(W, W_shape, W_init)
        self.b = self.resolve_param(b, b_shape, W_init)

    def calc_output_shape(self,
                          incoming,
                          filter_size,
                          num_filters,
                          num_channels,
                          inner_strides,
                          pad):

        input_shape = incoming.output_shape
        output_height = conv_output_length(input_shape[1],
                                           filter_size[0],
                                           inner_strides[0],
                                           pad)
        output_width = conv_output_length(input_shape[2],
                                           filter_size[1],
                                           inner_strides[1],
                                           pad)
        return (None,
                output_height, output_width,
                num_filters)

    def get_output(self, incoming_var):
        conv = tf.nn.conv2d(incoming_var,
                            self.W,
                            strides=self.strides,
                            padding=self.pad)
        return self.nonlin(conv + self.b)


def pool_output_length(input_length, pool_length, stride, pad):
    if pad == 'VALID':
        output_length = input_length + 2 * pad - pool_length + 1
        output_length = (output_length + stride - 1) // stride

    elif pad == 'SAME':
        if stride >= pool_length:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_length + stride - 1) // stride) + 1

    else:
        raise Exception('Invalid padding algorithm %s' % pad)

    return output_length


class MaxPool2DLayer(Layer):

    def __init__(self,
                 incoming,
                 pool_size,
                 inner_strides,
                 pad='SAME'):
        self.incoming = [incoming]
        self.output_shape = self.calc_output_shape(incoming,
                                                   pool_size,
                                                   inner_strides,
                                                   pad)
        print('pool_output_shape: %s' % str(self.output_shape))
        self.pool_size = (1,) + pool_size + (1,)
        self.strides = (1,) + inner_strides + (1,)
        self.pad = pad

    def calc_output_shape(self,
                          incoming,
                          pool_size,
                          inner_strides,
                          pad):
        input_shape = incoming.output_shape
        num_channels = incoming.output_shape[3]
        output_height = pool_output_length(input_shape[1],
                                           pool_size[0],
                                           inner_strides[0],
                                           pad)
        output_width = pool_output_length(input_shape[2],
                                          pool_size[0],
                                          inner_strides[0],
                                          pad)
        return (None, output_height, output_width, num_channels)

    def get_output(self, incoming_var):
        return tf.nn.max_pool(incoming_var,
                              ksize=self.pool_size,
                              strides=self.strides,
                              padding=self.pad)


class DropoutLayer(Layer):

    def __init__(self, incoming, keep_prob):
        self.incoming = [incoming]
        self.output_shape = incoming.output_shape
        self.keep_prob = keep_prob
        self.prob_var = tf.placeholder(tf.float32)

    def get_output(self, incoming_var):
        return tf.nn.dropout(incoming_var, self.prob_var)

    def get_supp_train_feed_dict(self):
        return {self.prob_var: self.keep_prob}

    def get_supp_test_feed_dict(self):
        return {self.prob_var: 1.0}
