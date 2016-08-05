import tensorflow as tf

from tfbrain import nonlin, init


class Layer(object):

    def __init__(self, incoming_list, name=None, trainable=True):
        self.incoming = incoming_list
        self.trainable = trainable
        self.chosen_name = name
        self.gen_name(0)
        self.params = {}
        self.config = {'name': name}

    def get_params(self):
        return self.params

    def get_trainable_params(self):
        if self.trainable:
            return self.params
        else:
            return {}

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

    def get_name(self):
        return self.name

    def get_output(self, incoming_var, **kwargs):
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
            return tf.Variable(init(expected_shape),
                               trainable=self.trainable)
        else:
            assert tuple(param.get_shape().as_list()) == expected_shape
            return param

    def resolve_param_pair(self,
                           W, W_shape,
                           b, b_shape,
                           W_b_init):
        if W or b is None:
            W, b = W_b_init(W_shape, b_shape)
            W = tf.Variable(W, trainable=self.trainable)
            b = tf.Variable(b, trainable=self.trainable)
        else:
            assert tuple(W.get_shape().as_list()) == W_shape
            assert tuple(b.get_shape().as_list()) == b_shape
        return W, b

    def get_input_hidden_var(self, **kwargs):
        return None

    def get_output_hidden_var(self, **kwargs):
        return None

    def get_init_hidden(self):
        return None

    def get_assign_hidden_op(self, **kwargs):
        return None


class InputLayer(Layer):

    def __init__(self,
                 shape,
                 dtype=tf.float32,
                 passthrough=None,
                 **kwargs):
        Layer.__init__(self, [], **kwargs)
        if passthrough is None:
            self.dtype = dtype
            self.output_shape = shape
            self.initialize_placeholder()
        else:
            self.placeholder = passthrough
            self.output_shape = passthrough.get_shape().as_list()
            print('passthrough output shape')
            print(self.output_shape)
        self.config.update({'shape': shape,
                            'dtype': dtype})

    def get_base_name(self):
        return 'input'

    def initialize_placeholder(self):
        self.placeholder = tf.placeholder(self.dtype,
                                          shape=self.output_shape,
                                          name=self.name)

    def get_output(self, incoming_var, **kwargs):
        if incoming_var is None:
            print('%s using placeholder' % self.get_name())
            return self.placeholder
        else:
            return incoming_var


class FullyConnectedLayer(Layer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin=nonlin.relu,
                 W_init=init.truncated_normal(),
                 b_init=init.constant(),
                 W_b_init=None,
                 W=None,
                 b=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.check_compatible(incoming)
        self.incoming_shape = incoming.get_output_shape()

        self.num_nodes = num_nodes
        self.nonlin = nonlin

        self.initialize_params(W, b, W_init, b_init, W_b_init)
        self.params = {'W': self.W,
                       'b': self.b}
        self.config.update({'num_nodes': num_nodes,
                            'nonlin': nonlin,
                            'W_init': W_init,
                            'b_init': b_init})

        self.output_shape = (None, num_nodes)

    def get_base_name(self):
        return 'fc'

    def check_compatible(self, incoming):
        if not len(incoming.get_output_shape()) == 2:
            raise Exception(('Incoming layer\'s output shape %s '
                             'incompatible, try passing it through '
                             'a FlattenLayer first')
                            % str(incoming.get_output_shape()))

    def initialize_params(self, W, b, W_init, b_init, W_b_init):
        W_shape = (self.incoming_shape[1], self.num_nodes)
        b_shape = (self.num_nodes,)

        if W_b_init is not None:
            self.W, self.b = self.resolve_param_pair(W, W_shape,
                                                     b, b_shape,
                                                     W_b_init)
        else:
            self.W = self.resolve_param(W, W_shape, W_init)
            self.b = self.resolve_param(b, b_shape, b_init)

    def get_output(self, incoming_var, **kwargs):
        return self.nonlin(
            tf.matmul(incoming_var, self.W) + self.b)
