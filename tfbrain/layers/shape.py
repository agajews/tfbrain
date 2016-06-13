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


class SliceLayer(Layer):

    def __init__(self,
                 incoming,
                 col=-1,
                 axis=1,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        incoming_shape = incoming.output_shape
        self.incoming_shape = incoming_shape
        self.output_shape = incoming_shape[:axis] + \
            incoming_shape[axis + 1:]
        self.col = col
        self.axis = axis

    def get_base_name(self):
        return 'slice'

    def get_output(self, incoming_var):
        begin = [0] * len(self.incoming_shape)
        begin[self.axis] = self.col
        size = [-1] * len(self.incoming_shape)
        size[self.axis] = 1
        return tf.slice(incoming_var, begin, size)


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
