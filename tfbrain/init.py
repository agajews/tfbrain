import tensorflow as tf


def zeros():
    return lambda s: tf.zeros(shape=s)


def truncated_normal(stddev=0.01, mean=0.0):
    return lambda s: tf.truncated_normal(shape=s,
                                         stddev=stddev,
                                         mean=mean)


def constant(val=0.1):
    return lambda s: tf.constant(val, shape=s)
