import tensorflow as tf
import numpy as np
import math


def zeros():
    return lambda s: tf.zeros(shape=s)


def truncated_normal(stddev=0.01, mean=0.0):
    return lambda s: tf.truncated_normal(shape=s,
                                         stddev=stddev,
                                         mean=mean)


def constant(val=0.1):
    return lambda s: tf.constant(val, shape=s)


def dqn_weight():
    def get_init(shape):
        print('dqn_weights')
        stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
        print(stdv)
        return tf.random_uniform(shape, minval=-stdv, maxval=stdv)
    return get_init


def dqn_bias():
    def get_init(shape):
        print('dqn_bias')
        stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
        return tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv)
    return get_init
