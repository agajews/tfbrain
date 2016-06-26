import tensorflow as tf

from tfbrain import nonlin, init


class Layer(object):

    def __init__(self, incoming_list, name=None):
        self.incoming = incoming_list
        self.chosen_name = name
        self.gen_name(0)
        self.params = {}

    def get_output_shape(self):
        return self.output_shape

    def get_base_name(self):
        return 'layer'

    def gen_name(self, num_global_layers):
        num_local_layers = 0
        for incoming in self.incoming:
            incoming.gen_name(num_global_layers)
            num_local_layers += incoming.num_local_layers
            num_global_layers += incoming.num_local_layers
        num_local_layers += 1  # this layer
        self.layer_num = num_global_layers + 1
        self.num_local_layers = num_local_layers
        if self.chosen_name is None:
            self.name = '%s_l%d' % (self.get_base_name(), self.layer_num)
        else:
            self.name = self.chosen_name

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

    def __init__(self,
                 shape,
                 dtype=tf.float32,
                 **kwargs):
        Layer.__init__(self, [], **kwargs)
        self.dtype = dtype
        self.output_shape = shape
        self.initialize_placeholder()

    def get_base_name(self):
        return 'input'

    def initialize_placeholder(self):
        self.placeholder = tf.placeholder(self.dtype,
                                          shape=self.output_shape)

    def get_output(self, incoming_var):
        return self.placeholder


class FullyConnectedLayer(Layer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin=nonlin.relu,
                 W_init=init.truncated_normal(),
                 b_init=init.constant(),
                 W=None,
                 b=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.check_compatible(incoming)
        self.incoming_shape = incoming.get_output_shape()

        self.num_nodes = num_nodes
        self.nonlin = nonlin

        self.initialize_params(W, b, W_init, b_init)
        self.params = {'W': self.W,
                       'b': self.b}

        self.output_shape = (None, num_nodes)

    def get_base_name(self):
        return 'fc'

    def check_compatible(self, incoming):
        if not len(incoming.get_output_shape()) == 2:
            raise Exception(('Incoming layer\'s output shape %s '
                             'incompatible, try passing it through '
                             'a FlattenLayer first')
                            % str(incoming.get_output_shape()))

    def initialize_params(self, W, b, W_init, b_init):
        W_shape = (self.incoming_shape[1], self.num_nodes)
        b_shape = (self.num_nodes,)

        self.W = self.resolve_param(W, W_shape, W_init)
        self.b = self.resolve_param(b, b_shape, b_init)

    def get_output(self, incoming_var):
        return self.nonlin(
            tf.matmul(incoming_var, self.W) + self.b)
