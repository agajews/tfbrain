import tensorflow as tf

from .core import Layer


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
                 pad='SAME',
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)

        self.output_shape = self.calc_output_shape(incoming,
                                                   pool_size,
                                                   inner_strides,
                                                   pad)
        self.pool_size = (1,) + pool_size + (1,)
        self.strides = (1,) + inner_strides + (1,)
        self.pad = pad
        self.config.update({'pool_size': pool_size,
                            'inner_strides': inner_strides,
                            'pad': pad})

    def get_base_name(self):
        return 'pool2d'

    def calc_output_shape(self,
                          incoming,
                          pool_size,
                          inner_strides,
                          pad):
        input_shape = incoming.get_output_shape()
        num_channels = incoming.get_output_shape()[3]
        output_height = pool_output_length(input_shape[1],
                                           pool_size[0],
                                           inner_strides[0],
                                           pad)
        output_width = pool_output_length(input_shape[2],
                                          pool_size[0],
                                          inner_strides[0],
                                          pad)
        return (None, output_height, output_width, num_channels)

    def get_output(self, incoming_var, **kwargs):
        return tf.nn.max_pool(incoming_var,
                              ksize=self.pool_size,
                              strides=self.strides,
                              padding=self.pad)
