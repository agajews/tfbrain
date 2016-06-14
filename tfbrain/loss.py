import tensorflow as tf


class Loss(object):

    def __init__(self, hyperparams):
        '''hyperparams: a dictionary of hyperparameters'''
        self.hyperparams = hyperparams

    def build(self, y_hat, y):
        '''y_hat: a TF tensor representing a model's output
        y: a TF placeholder representing the expected output'''
        raise NotImplementedError()


class Crossentropy(Loss):

    def build(self, y_hat, y):
        return tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(y_hat),
                           reduction_indices=[1]))
