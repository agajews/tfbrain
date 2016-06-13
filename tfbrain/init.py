import tensorflow as tf


def zeros(shape):
    return tf.zeros(shape=shape)


def truncated_normal(shape):
    return tf.truncated_normal(shape, stddev=0.1)


def constant(shape):
    return tf.constant(0.1, shape=shape)
