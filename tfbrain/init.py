import tensorflow as tf


def zeros(shape):
    return tf.zeros(shape=shape)


def truncated_normal(shape, stddev=0.1):
    return tf.truncated_normal(shape, stddev=stddev)


def constant(shape, val=0.1):
    return tf.constant(val, shape=shape)
