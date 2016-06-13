import tensorflow as tf


class Optimizer(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def get_train_step(self, loss_val):
        raise NotImplementedError()


class SGDOptim(Optimizer):

    def get_train_step(self, loss_val):
        return tf.train.GradientDescentOptimizer(
            self.hyperparams['learning_rate']).minimize(loss_val)


class AdamOptim(Optimizer):

    def get_train_step(self, loss_val):
        return tf.train.AdamOptimizer(
            self.hyperparams['learning_rate']).minimize(loss_val)
