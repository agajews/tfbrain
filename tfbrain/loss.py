import tensorflow as tf

from tfbrain.helpers import create_x_feed_dict, \
    create_y_feed_dict, create_supp_test_feed_dict


class Loss(object):

    def __init__(self, hyperparams):
        '''hyperparams: a dictionary of hyperparameters'''
        self.hyperparams = hyperparams

    def build(self, y_hat, y):
        '''y_hat: a TF tensor representing a model's output
        y: a TF placeholder representing the expected output'''
        raise NotImplementedError()

    def compute(self, model, xs, y_var, y_val):
        feed_dict = create_x_feed_dict(model.input_vars, xs)
        feed_dict.update(create_y_feed_dict(y_var, y_val))
        feed_dict.update(create_supp_test_feed_dict(model))
        loss = self.loss.eval(feed_dict=feed_dict)
        return loss


class MSE(Loss):

    def build(self, y_hat, y, mask=None):
        if mask is None:
            self.loss = tf.reduce_mean(
                tf.square(y - y_hat))
        else:
            self.loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(mask * (y - y_hat)),
                              reduction_indices=[1]))

        return self.loss


class Crossentropy(Loss):

    def build(self, y_hat, y, mask=None):
        if mask is None:
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_hat, 1e-10, 1.0)),
                               reduction_indices=[1]))
        else:
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(
                    mask * y * tf.log(tf.clip_by_value(y_hat, 1e-10, 1.0)),
                    reduction_indices=[1]))
        return self.loss
