import tensorflow as tf

from functools import reduce

from .core import Layer


class FlattenLayer(Layer):

    def __init__(self,
                 incoming,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.output_shape = self.flatten_shape(incoming.output_shape)

    def get_base_name(self):
        return 'flat'

    def flatten_shape(self, shape):
        return (None,
                reduce(lambda x, y: x * y,
                       shape[1:]))

    def get_output(self, incoming_var):
        return tf.reshape(incoming_var, (-1,) + self.output_shape[1:])


class ReshapeLayer(Layer):

    def __init__(self,
                 incoming,
                 shape,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.output_shape = list(map(lambda d: d if d is not None else -1,
                                     shape))

    def get_base_name(self):
        return 'reshape'

    def get_output(self, incoming_var):
        return tf.reshape(incoming_var, self.output_shape)


class SeqSliceLayer(Layer):

    def __init__(self,
                 incoming,
                 col=-1,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        incoming_shape = incoming.output_shape
        self.incoming_shape = incoming_shape
        self.output_shape = incoming_shape[:1] + \
            incoming_shape[2:]
        self.check_compatible(incoming)
        self.col = col

    def check_compatible(self, incoming):
        if not len(incoming.output_shape) == 3:
            raise Exception(('Incoming layer\'s output shape %s '
                             'incompatible, this is only for sequences')
                            % str(incoming.output_shape))

    def get_base_name(self):
        return 'slice'

    def get_output(self, incoming_var):
        if self.col == -1:
            return tf.reverse(incoming_var, [False, True, False])[:, 0, :]
        else:
            return incoming_var[:, self.col, :]


class MergeLayer(Layer):

    def __init__(self,
                 incoming_list,
                 axis=1,
                 **kwargs):
        Layer.__init__(self, incoming_list, **kwargs)
        self.axis = axis
        self.output_shape = self.calc_output_shape()

    def get_base_name(self):
        return 'merge'

    def calc_output_shape(self):
        output_shape = list(self.incoming[0].output_shape)
        output_shape[self.axis] = -1
        axis_length = self.incoming[0].output_shape[self.axis]
        for incoming in self.incoming[1:]:
            self.check_compatible(incoming, output_shape)
            axis_length += incoming.output_shape[self.axis]

        output_shape[self.axis] = axis_length
        return tuple(output_shape)

    def check_compatible(self, incoming, output_shape):
        incoming_shape = list(incoming.output_shape)
        incoming_shape[self.axis] = -1
        assert incoming_shape == output_shape

    def get_output(self, *incoming_vars):
        return tf.concat(self.axis, incoming_vars)
