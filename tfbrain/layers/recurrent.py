import tensorflow as tf

from tfbrain import nonlin
from .core import Layer


class BasicRNNLayer(Layer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin=nonlin.relu,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.incoming_shape = incoming.output_shape
        self.num_nodes = num_nodes
        self.check_compatible(incoming)
        self.output_shape = self.calc_output_shape()

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

    def get_output(self, incoming_var):
        cell = tf.nn.rnn_cell.BasicRNNCell(self.num_nodes)
        return tf.nn.rnn.dynamic_rnn(cell, incoming_var)
