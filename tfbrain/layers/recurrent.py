import tensorflow as tf

from tfbrain import nonlin, init
from tfbrain.helpers import get_output, get_input_name
from .core import Layer


class BasicRNNLayer(Layer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin_h=nonlin.relu,
                 nonlin_o=nonlin.relu,
                 W_init=init.truncated_normal(),
                 b_init=init.constant(),
                 W_b_init=None,
                 W_h=None,
                 W_i=None,
                 W_o=None,
                 b_h=None,
                 b_o=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.incoming_shape = incoming.get_output_shape()
        self.num_nodes = num_nodes
        self.nonlin_o = nonlin_o
        self.nonlin_h = nonlin_h
        self.check_compatible(incoming)
        self.output_shape = self.calc_output_shape()

        self.initialize_params(W_h, W_i, W_o,
                               b_h, b_o,
                               W_init, b_init, W_b_init)

        self.params = {'W_h': self.W_h,
                       'W_i': self.W_i,
                       'W_o': self.W_o,
                       'b_h': self.b_o,
                       'b_o': self.b_h}

        self.input_hidden = None
        self.input_timestep_hidden = None
        self.timestep_hidden = None
        self.config.update({
            'num_nodes': num_nodes,
            'nonlin_h': nonlin_h,
            'nonlin_o': nonlin_o,
            'W_init': W_init,
            'b_init': b_init,
            'W_b_init': W_b_init
        })

    def initialize_params(self,
                          W_h, W_i, W_o,
                          b_h, b_o,
                          W_init, b_init, W_b_init):

        W_h_shape = (self.num_nodes, self.num_nodes)
        W_i_shape = (self.incoming_shape[2], self.num_nodes)
        W_o_shape = (self.num_nodes, self.num_nodes)
        b_h_shape = (self.num_nodes,)
        b_o_shape = (self.num_nodes,)

        if W_b_init is not None:
            self.W_h, self.b_h = self.resolve_param_pair(W_h, W_h_shape,
                                                         b_h, b_h_shape,
                                                         W_b_init)

            self.W_o, self.b_o = self.resolve_param_pair(W_o, W_o_shape,
                                                         b_o, b_o_shape,
                                                         W_b_init)

            self.W_i = self.resolve_param(W_i, W_i_shape, W_init)
        else:
            self.W_h = self.resolve_param(W_h, W_h_shape, W_init)
            self.W_i = self.resolve_param(W_i, W_i_shape, W_init)
            self.W_o = self.resolve_param(W_o, W_o_shape, W_init)
            self.b_h = self.resolve_param(b_h, b_h_shape, b_init)
            self.b_o = self.resolve_param(b_o, b_o_shape, b_init)

    def calc_output_shape(self):
        return self.incoming_shape[:2] + (self.num_nodes,)

    def check_compatible(self, incoming):
        if not len(incoming.get_output_shape()) == 3:
            raise Exception(('Incoming layer\'s output shape %s '
                             'incompatible, try passing it through '
                             'a ReshapeLayer or SequenceFlattenLayer first')
                            % str(incoming.get_output_shape()))

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

    # def get_single_hidden_shape(self):
    #     return (1, self.num_nodes)

    def get_input_hidden_var(self, timestep=False):
        if not timestep:
            return self.input_hidden
        else:
            return self.input_timestep_hidden

    def get_output_hidden_var(self):
        return self.output_timestep_hidden

    def get_init_hidden(self):
        return self.init_hidden

    def get_assign_hidden_op(self, zero=False, **kwargs):
        if zero:
            return self.zero_hidden_op
        else:
            return self.assign_hidden_op

    def get_output(self, incoming_var,
                   timestep=False,
                   input_hidden=False,
                   **kwargs):
        if not timestep:
            if not input_hidden:
                initial_hidden = self.get_initial_hidden(incoming_var)
            else:
                self.input_hidden = tf.placeholder(
                    tf.float32,
                    self.get_initial_hidden(incoming_var).get_shape())
                initial_hidden = self.input_hidden

            incoming_var = tf.transpose(incoming_var, (1, 0, 2))
            states = tf.scan(self.recurrence_fn,
                             incoming_var,
                             initializer=initial_hidden)

            outputs = tf.map_fn(self.output_fn,
                                states)

            outputs = tf.transpose(outputs, (1, 0, 2), name='end_transpose')
            return outputs

        if input_hidden:
            self.input_timestep_hidden = tf.placeholder(
                tf.float32,
                shape=self.get_initial_hidden(incoming_var).get_shape(),
                name='timestep_hidden')
            self.init_hidden = self.get_initial_hidden(incoming_var)
            hidden = self.input_timestep_hidden
        else:
            if self.timestep_hidden is None:
                self.timestep_hidden = self.get_initial_hidden(incoming_var)
            hidden = self.timestep_hidden
        state = self.recurrence_fn(hidden, incoming_var[:, 0, :])
        if input_hidden:
            self.output_timestep_hidden = state
        else:
            self.timestep_hidden = state
        output = self.output_fn(state)
        output = tf.expand_dims(output, 1)
        return output


class LSTMLayer(BasicRNNLayer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin_i=nonlin.sigmoid,
                 nonlin_c=nonlin.tanh,
                 nonlin_o=nonlin.relu,
                 W_init=init.truncated_normal(),
                 b_init=init.constant(),
                 U_init=init.truncated_normal(),
                 W_b_init=None,
                 W_i=None,
                 U_i=None,
                 W_f=None,
                 U_f=None,
                 W_g=None,
                 U_g=None,
                 W_c=None,
                 U_c=None,
                 W_o=None,
                 b_i=None,
                 b_f=None,
                 b_g=None,
                 b_c=None,
                 b_o=None,
                 **kwargs):

        Layer.__init__(self, [incoming], **kwargs)
        self.incoming_shape = incoming.get_output_shape()
        self.num_nodes = num_nodes
        self.nonlin_i = nonlin_i
        self.nonlin_c = nonlin_c
        self.nonlin_o = nonlin_o
        self.check_compatible(incoming)
        self.output_shape = self.calc_output_shape()

        self.initialize_params(W_i, U_i, W_f, U_f, W_g, U_g,
                               W_c, U_c, W_o,
                               b_i, b_f, b_g, b_c, b_o,
                               W_init, b_init, W_b_init, U_init)

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

        self.config.update({
            'num_nodes': num_nodes,
            'nonlin_i': nonlin_i,
            'nonlin_c': nonlin_c,
            'nonlin_o': nonlin_o,
            'W_init': W_init,
            'b_init': b_init,
            'W_b_init': W_b_init
        })

        self.input_hidden = None
        self.input_timestep_hidden = None
        self.timestep_hidden = None

    def initialize_params(self,
                          W_i, U_i, W_f, U_f, W_g, U_g,
                          W_c, U_c, W_o,
                          b_i, b_f, b_g, b_c, b_o,
                          W_init, b_init, W_b_init, U_init):

        input_size = self.incoming_shape[2]

        W_i_shape = (input_size, self.num_nodes)
        U_i_shape = (self.num_nodes, self.num_nodes)

        W_f_shape = (input_size, self.num_nodes)
        U_f_shape = (self.num_nodes, self.num_nodes)

        W_g_shape = (input_size, self.num_nodes)
        U_g_shape = (self.num_nodes, self.num_nodes)

        W_c_shape = (input_size, self.num_nodes)
        U_c_shape = (self.num_nodes, self.num_nodes)

        W_o_shape = (self.num_nodes, self.num_nodes)

        b_i_shape = (self.num_nodes,)
        b_f_shape = (self.num_nodes,)
        b_g_shape = (self.num_nodes,)
        b_c_shape = (self.num_nodes,)
        b_o_shape = (self.num_nodes,)

        if W_b_init is not None:
            self.W_i, self.b_i = self.resolve_param_pair(W_i, W_i_shape,
                                                         b_i, b_i_shape,
                                                         W_b_init)
            self.W_f, self.b_f = self.resolve_param_pair(W_f, W_f_shape,
                                                         b_f, b_f_shape,
                                                         W_b_init)
            self.W_g, self.b_g = self.resolve_param_pair(W_g, W_g_shape,
                                                         b_g, b_g_shape,
                                                         W_b_init)
            self.W_c, self.b_c = self.resolve_param_pair(W_c, W_c_shape,
                                                         b_c, b_c_shape,
                                                         W_b_init)
            self.W_o, self.b_o = self.resolve_param_pair(W_o, W_o_shape,
                                                         b_o, b_o_shape,
                                                         W_b_init)
        else:
            self.W_i = self.resolve_param(W_i, W_i_shape, W_init)
            self.b_i = self.resolve_param(b_i, b_i_shape, b_init)

            self.W_f = self.resolve_param(W_f, W_f_shape, W_init)
            self.b_f = self.resolve_param(b_f, b_f_shape, b_init)

            self.W_g = self.resolve_param(W_g, W_g_shape, W_init)
            self.b_g = self.resolve_param(b_g, b_g_shape, b_init)

            self.W_c = self.resolve_param(W_c, W_c_shape, W_init)
            self.b_c = self.resolve_param(b_c, b_c_shape, b_init)

            self.W_o = self.resolve_param(W_o, W_o_shape, W_init)
            self.b_o = self.resolve_param(b_o, b_o_shape, b_init)

        self.U_i = self.resolve_param(U_i, U_i_shape, U_init)
        self.U_f = self.resolve_param(U_f, U_f_shape, U_init)
        self.U_g = self.resolve_param(U_g, U_g_shape, U_init)
        self.U_c = self.resolve_param(U_c, U_c_shape, U_init)

    def get_base_name(self):
        return 'lstm'

    # def get_single_hidden_shape(self):
    #     return (2, 1, self.num_nodes)

    def get_initial_hidden(self, incoming_var):
        initial_hidden = tf.matmul(
            incoming_var[:, 0, :],
            tf.zeros([self.incoming_shape[2], self.num_nodes]))
        return tf.pack([initial_hidden, initial_hidden])

    def recurrence_fn(self, h_prev_tuple, x_t):
        h_prev, c_prev = tf.unpack(h_prev_tuple)

        i = self.nonlin_i(
            tf.matmul(x_t, self.W_i,
                      name='i_w') +
            tf.matmul(h_prev, self.U_i,
                      name='i_u') +
            self.b_i)

        f = self.nonlin_i(
            tf.matmul(x_t, self.W_f,
                      name='f_w') +
            tf.matmul(h_prev, self.U_f,
                      name='f_u') +
            self.b_f)

        g = self.nonlin_i(
            tf.matmul(x_t, self.W_g,
                      name='g_w') +
            tf.matmul(h_prev, self.U_g,
                      name='g_u') +
            self.b_g)

        c_tilda = self.nonlin_c(
            tf.matmul(x_t, self.W_c,
                      name='ct_w') +
            tf.matmul(h_prev, self.U_c,
                      name='ct_u') +
            self.b_c)

        c = f * c_prev + i * c_tilda

        h = g * self.nonlin_c(c)

        return tf.pack([h, c])

    def output_fn(self, states_tuple):
        h = states_tuple[0, :, :]
        return self.nonlin_o(
            tf.matmul(h, self.W_o) +
            self.b_o)


class NetOnSeq(Layer):

    def __init__(self,
                 incoming,
                 net,
                 **kwargs):

        Layer.__init__(self, [incoming], **kwargs)
        self.incoming_shape = incoming.get_output_shape()
        self.net = net
        self.output_shape = self.calc_output_shape()
        self.config.update({'net': net})

    def calc_output_shape(self):
        return (None, None) + self.net.get_output_shape()[1:]

    def get_base_name(self):
        return 'net_on_seq'

    def output_fn(self, state):
        return get_output(self.net, {get_input_name(self.net): state})

    def get_output(self, incoming_var, **kwargs):
        incoming_var = tf.transpose(
            incoming_var, (1, 0, *range(2, len(incoming_var.get_shape()))),
            name='front_transpose')

        outputs = tf.map_fn(self.output_fn,
                            incoming_var)

        outputs = tf.transpose(
            outputs, (1, 0, 2),
            name='end_transpose')
        return outputs
