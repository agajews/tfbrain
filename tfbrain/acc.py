import tensorflow as tf


class Acc(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def build(self):
        raise NotImplementedError()


class CatAcc(Acc):

    def build(self, y_hat, y):
        correct_prediction = tf.equal(tf.argmax(y_hat, 1),
                                      tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy


class Perplexity(Acc):

    def build(self, y_hat, y):
        return tf.exp(tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(y_hat),
                           reduction_indices=[1])))
