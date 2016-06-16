import tensorflow as tf

from tfbrain.helpers import create_x_feed_dict, \
    create_y_feed_dict, create_supp_test_feed_dict


class Acc(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def build(self):
        raise NotImplementedError()

    def compute(self, model, xs, y_var, y_val):
        feed_dict = create_x_feed_dict(model.input_vars, xs)
        feed_dict.update(create_y_feed_dict(y_var, y_val))
        feed_dict.update(create_supp_test_feed_dict(model))
        accuracy = self.accuracy.eval(feed_dict=feed_dict)
        return accuracy


class CatAcc(Acc):

    def build(self, y_hat, y):
        correct_prediction = tf.equal(tf.argmax(y_hat, 1),
                                      tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.accuracy


class Perplexity(Acc):

    def build(self, y_hat, y):
        self.accuracy = tf.exp(tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(y_hat),
                           reduction_indices=[1])))
        return self.accuracy
