import tensorflow as tf


def tanh(incoming_var):
    return tf.tanh(incoming_var)


def softmax(incoming_var):
    return tf.nn.softmax(incoming_var)


def relu(incoming_var):
    return tf.nn.relu(incoming_var)


def sigmoid(incoming_var):
    return tf.sigmoid(incoming_var)


def identity(incoming_var):
    return incoming_var
