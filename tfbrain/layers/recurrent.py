import tensorflow as tf

from tfbrain import nonlin, init
from .core import Layer


class BasicRNNLayer(Layer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin_h=nonlin.relu,
                 nonlin_o=nonlin.relu,
                 W_h_init=init.truncated_normal,
                 W_h=None,
                 W_i_init=init.truncated_normal,
                 W_i=None,
                 W_o_init=init.truncated_normal,
                 W_o=None,
                 b_h_init=init.constant,
                 b_h=None,
                 b_o_init=init.constant,
                 b_o=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.incoming_shape = incoming.output_shape
        self.num_nodes = num_nodes
        self.nonlin_o = nonlin_o
        self.nonlin_h = nonlin_h
        self.check_compatible(incoming)
        self.output_shape = self.calc_output_shape()

        self.initialize_params(W_h, W_i, W_o,
                               b_h, b_o,
                               W_h_init, W_i_init, W_o_init,
                               b_h_init, b_o_init)
        self.params = {'W_h': self.W_h,
                       'W_i': self.W_i,
                       'W_o': self.W_o,
                       'b_h': self.b_o,
                       'b_o': self.b_h}

    def initialize_params(self,
                          W_h, W_i, W_o,
                          b_h, b_o,
                          W_h_init, W_i_init, W_o_init,
                          b_h_init, b_o_init):
        W_h_shape = (self.num_nodes, self.num_nodes)
        W_i_shape = (self.incoming_shape[2], self.num_nodes)
        W_o_shape = (self.num_nodes, self.num_nodes)
        b_h_shape = (self.num_nodes,)
        b_o_shape = (self.num_nodes,)

        self.W_h = self.resolve_param(W_h, W_h_shape, W_h_init)
        self.W_i = self.resolve_param(W_i, W_i_shape, W_i_init)
        self.W_o = self.resolve_param(W_o, W_o_shape, W_o_init)
        self.b_h = self.resolve_param(b_h, b_h_shape, b_h_init)
        self.b_o = self.resolve_param(b_o, b_o_shape, b_o_init)

    def calc_output_shape(self):
        return self.incoming_shape[:2] + (self.num_nodes,)

    def check_compatible(self, incoming):
        if not len(incoming.output_shape) == 3:
            raise Exception(('Incoming layer\'s output shape %s '
                             'incompatible, try passing it through '
                             'a ReshapeLayer or SequenceFlattenLayer first')
                            % str(incoming.output_shape))

    def get_base_name(self):
        return 'brnn'

    def recurrence_fn(self, h_prev, x_t):
        return self.nonlin_h(
            tf.matmul(h_prev, self.W_h) +
            tf.matmul(x_t, self.W_i) + self.b_h)

    def output_fn(self, state):
        return self.nonlin_o(
            tf.matmul(state, self.W_o) + self.b_o)

    def get_initial_hidden(self, incoming_var):
        initial_hidden = tf.matmul(
            incoming_var[:, 0, :],
            tf.zeros([self.incoming_shape[2], self.num_nodes]))
        return initial_hidden

    def transform_states(self, states):
        return states

    def get_output(self, incoming_var):
        initial_hidden = self.get_initial_hidden(incoming_var)
        incoming_var = tf.transpose(incoming_var, (1, 0, 2))
        states = tf.scan(self.recurrence_fn,
                         incoming_var,
                         initializer=initial_hidden)

        states = self.transform_states(states)
        outputs = tf.map_fn(self.output_fn,
                            states)

        outputs = tf.transpose(outputs, (1, 0, 2), name='end_transpose')
        return outputs


class LSTMLayer(BasicRNNLayer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin_i=nonlin.sigmoid,
                 nonlin_c=nonlin.tanh,
                 nonlin_o=nonlin.relu,
                 W_i_init=init.truncated_normal,
                 W_i=None,
                 U_i_init=init.truncated_normal,
                 U_i=None,
                 W_f_init=init.truncated_normal,
                 W_f=None,
                 U_f_init=init.truncated_normal,
                 U_f=None,
                 W_g_init=init.truncated_normal,
                 W_g=None,
                 U_g_init=init.truncated_normal,
                 U_g=None,
                 W_c_init=init.truncated_normal,
                 W_c=None,
                 U_c_init=init.truncated_normal,
                 U_c=None,
                 W_o_init=init.truncated_normal,
                 W_o=None,
                 b_i_init=init.constant,
                 b_i=None,
                 b_f_init=init.constant,
                 b_f=None,
                 b_g_init=init.constant,
                 b_g=None,
                 b_c_init=init.constant,
                 b_c=None,
                 b_o_init=init.constant,
                 b_o=None,
                 **kwargs):

        Layer.__init__(self, [incoming], **kwargs)
        self.incoming_shape = incoming.output_shape
        self.num_nodes = num_nodes
        self.nonlin_i = nonlin_i
        self.nonlin_c = nonlin_c
        self.nonlin_o = nonlin_o
        self.check_compatible(incoming)
        self.output_shape = self.calc_output_shape()

        self.initialize_params(W_i, U_i, W_f, U_f, W_g, U_g,
                               W_c, U_c, W_o,
                               b_i, b_f, b_g, b_c, b_o,
                               W_i_init, U_i_init,
                               W_f_init, U_f_init,
                               W_g_init, U_g_init,
                               W_c_init, U_c_init,
                               W_o_init,
                               b_i_init, b_f_init,
                               b_g_init, b_c_init,
                               b_o_init)

        self.params = {'W_i': self.W_i,
                       'U_i': self.U_i,
                       'W_f': self.W_f,
                       'U_f': self.U_f,
                       'W_g': self.W_g,
                       'U_g': self.U_g,
                       'W_c': self.W_c,
                       'U_c': self.U_c,
                       'W_o': self.W_o,
                       'b_i': self.b_i,
                       'b_f': self.b_f,
                       'b_g': self.b_g,
                       'b_c': self.b_c,
                       'b_o': self.b_o}

    def initialize_params(self,
                          W_i, U_i, W_f, U_f, W_g, U_g,
                          W_c, U_c, W_o,
                          b_i, b_f, b_g, b_c, b_o,
                          W_i_init, U_i_init,
                          W_f_init, U_f_init,
                          W_g_init, U_g_init,
                          W_c_init, U_c_init,
                          W_o_init,
                          b_i_init, b_f_init,
                          b_g_init, b_c_init,
                          b_o_init):

        input_size = self.incoming_shape[2]

        W_i_shape = (input_size, self.num_nodes)
        U_i_shape = (self.num_nodes, self.num_nodes)

        W_f_shape = (input_size, self.num_nodes)
        U_f_shape = (self.num_nodes, self.num_nodes)

        W_g_shape = (self.num_nodes, self.num_nodes)
        U_g_shape = (self.num_nodes, self.num_nodes)

        W_c_shape = (input_size, self.num_nodes)
        U_c_shape = (self.num_nodes, self.num_nodes)

        W_o_shape = (self.num_nodes, self.num_nodes)

        b_i_shape = (self.num_nodes,)
        b_f_shape = (self.num_nodes,)
        b_g_shape = (self.num_nodes,)
        b_c_shape = (self.num_nodes,)
        b_o_shape = (self.num_nodes,)

        self.W_i = self.resolve_param(W_i, W_i_shape, W_i_init)
        self.U_i = self.resolve_param(U_i, U_i_shape, U_i_init)

        self.W_f = self.resolve_param(W_f, W_f_shape, W_f_init)
        self.U_f = self.resolve_param(U_f, U_f_shape, U_f_init)

        self.W_g = self.resolve_param(W_g, W_g_shape, W_g_init)
        self.U_g = self.resolve_param(U_g, U_g_shape, U_g_init)

        self.W_c = self.resolve_param(W_c, W_c_shape, W_c_init)
        self.U_c = self.resolve_param(U_c, U_c_shape, U_c_init)

        self.W_o = self.resolve_param(W_o, W_o_shape, W_o_init)

        self.b_i = self.resolve_param(b_i, b_i_shape, b_i_init)
        self.b_f = self.resolve_param(b_f, b_f_shape, b_f_init)
        self.b_g = self.resolve_param(b_g, b_g_shape, b_g_init)
        self.b_c = self.resolve_param(b_c, b_c_shape, b_c_init)
        self.b_o = self.resolve_param(b_o, b_o_shape, b_o_init)

    def get_base_name(self):
        return 'lstm'

    def get_initial_hidden(self, incoming_var):
        initial_hidden = tf.matmul(
            incoming_var[:, 0, :],
            tf.zeros([self.incoming_shape[2], self.num_nodes]))
        return tf.pack([initial_hidden, initial_hidden])

    def recurrence_fn(self, h_prev_tuple, x_t):
        h_prev, c_prev = tf.unpack(h_prev_tuple)

        i = self.nonlin_i(
            tf.matmul(x_t, self.W_i) +
            tf.matmul(h_prev, self.U_i) +
            self.b_i)

        f = self.nonlin_i(
            tf.matmul(x_t, self.W_f) +
            tf.matmul(h_prev, self.U_f) +
            self.b_f)

        g = self.nonlin_i(
            tf.matmul(x_t, self.W_g) +
            tf.matmul(h_prev, self.U_g) +
            self.b_g)

        c_tilda = self.nonlin_c(
            tf.matmul(x_t, self.W_c) +
            tf.matmul(h_prev, self.U_c) +
            self.b_c)

        c = f * c_prev + i * c_tilda

        h = g * self.nonlin_c(c)

        return tf.pack([h, c])

    def transform_states(self, states):
        return states[:, 0, :, :]

    def output_fn(self, h):
        return self.nonlin_o(
            tf.matmul(h, self.W_o) +
            self.b_o)
