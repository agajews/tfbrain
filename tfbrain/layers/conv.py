import tensorflow as tf

from tfbrain import nonlin, init
from .core import Layer


def conv_output_length(input_length,
                       filter_length,
                       stride,
                       pad):
    if pad == 'VALID':
        output_length = input_length - filter_length + 1
    elif pad == 'SAME':
        output_length = input_length
    else:
        raise Exception('Invalid padding algorithm %s' % pad)

    return (output_length + stride - 1) // stride


class Conv2DLayer(Layer):

    def __init__(self,
                 incoming,
                 filter_size,
                 num_filters,
                 inner_strides=(1, 1),
                 pad='SAME',
                 nonlin=nonlin.relu,
                 W_init=init.truncated_normal(),
                 b_init=init.constant(),
                 W_b_init=None,
                 W=None,
                 b=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)

        self.nonlin = nonlin
        num_channels = incoming.get_output_shape()[3]
        self.output_shape = self.calc_output_shape(incoming,
                                                   filter_size,
                                                   num_filters,
                                                   num_channels,
                                                   inner_strides,
                                                   pad)
        print(self.output_shape)

        self.strides = (1,) + inner_strides + (1,)
        self.pad = pad

        W_shape = filter_size + (num_channels, num_filters)
        b_shape = (num_filters,)

        if W_b_init is not None:
            self.W, self.b = self.resolve_param_pair(W, W_shape,
                                                     b, b_shape,
                                                     W_b_init)
        else:
            self.W = self.resolve_param(W, W_shape, W_init)
            self.b = self.resolve_param(b, b_shape, b_init)

        print(self.b.get_shape())
        # self.W = self.resolve_param(W, W_shape, W_init)
        # self.b = self.resolve_param(b, b_shape, b_init)
        self.params = {'W': self.W,
                       'b': self.b}
        self.config.update({'filter_size': filter_size,
                            'num_filters': num_filters,
                            'inner_strides': inner_strides,
                            'pad': pad,
                            'nonlin': nonlin,
                            'W_init': W_init,
                            'b_init': b_init,
                            'W_b_init': W_b_init})

    def get_base_name(self):
        return 'conv2d'

    def calc_output_shape(self,
                          incoming,
                          filter_size,
                          num_filters,
                          num_channels,
                          inner_strides,
                          pad):

        input_shape = incoming.get_output_shape()
        output_height = conv_output_length(input_shape[1],
                                           filter_size[0],
                                           inner_strides[0],
                                           pad)
        output_width = conv_output_length(input_shape[2],
                                          filter_size[1],
                                          inner_strides[1],
                                          pad)
        return (None,
                output_height, output_width,
                num_filters)

    def get_output(self, incoming_var, **kwargs):
        conv = tf.nn.conv2d(incoming_var,
                            self.W,
                            strides=self.strides,
                            padding=self.pad)
        return self.nonlin(conv + self.b)
