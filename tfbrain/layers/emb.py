import tensorflow as tf

from tfbrain import init
from .core import Layer


class EmbeddingLayer(Layer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 num_cats,
                 E_init=init.truncated_normal(),
                 E=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.num_nodes = num_nodes
        self.output_shape = incoming.get_output_shape() + (num_nodes,)
        E_shape = (num_cats, num_nodes)
        self.E = self.resolve_param(E, E_shape, E_init)
        self.params = {'E': self.E}

    def get_base_name(self):
        return 'emb'

    def get_output(self, incoming_var):
        return tf.nn.embedding_lookup(self.E, incoming_var)
