import tensorflow as tf

from .core import Layer


class DropoutLayer(Layer):

    def __init__(self,
                 incoming,
                 keep_prob,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.output_shape = incoming.get_output_shape()
        self.keep_prob = keep_prob
        self.prob_var = tf.placeholder(tf.float32)

    def get_base_name(self):
        return 'drop'

    def get_output(self, incoming_var):
        return tf.nn.dropout(incoming_var, self.prob_var)

    def get_supp_train_feed_dict(self):
        return {self.prob_var: self.keep_prob}

    def get_supp_test_feed_dict(self):
        return {self.prob_var: 1.0}
