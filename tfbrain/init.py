import tensorflow as tf
import numpy as np
import math


def zeros():
    def get_init(shape, partner=0):
        return tf.zeros(shape=shape)
    return get_init


def truncated_normal(stddev=0.01, mean=0.0):
    def get_init(shape):
        return tf.truncated_normal(shape=shape,
                                   stddev=stddev,
                                   mean=mean)
    return get_init


def constant(val=0.1):
    def get_init(shape):
        return tf.constant(val, shape=shape)
    return get_init


def dqn():
    def get_init(W_shape, b_shape):
        print('dqn_weights')
        stdv = 1.0 / math.sqrt(np.prod(W_shape[0:-1]))
        print(stdv)
        W = tf.random_uniform(W_shape, minval=-stdv, maxval=stdv)
        b = tf.random_uniform(b_shape, minval=-stdv, maxval=stdv)
        return W, b
    return get_init
