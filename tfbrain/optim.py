import tensorflow as tf


class Optimizer(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def get_train_step(self, loss_val):
        raise NotImplementedError()


class SGDOptim(Optimizer):

    def get_train_step(self, loss_val):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss_val, tvars),
                                          self.hyperparams['grad_norm_clip'])
        optimizer = tf.train.GradientDescentOptimizer(
            self.hyperparams['learning_rate'])
        return optimizer.apply_gradients(zip(grads, tvars))


class AdamOptim(Optimizer):

    def get_train_step(self, loss_val):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss_val, tvars),
                                          self.hyperparams['grad_norm_clip'])
        optimizer = tf.train.AdamOptimizer(
            self.hyperparams['learning_rate'])
        return optimizer.apply_gradients(zip(grads, tvars))
