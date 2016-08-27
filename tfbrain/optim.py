import tensorflow as tf


class Optimizer(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def get_train_step(self, loss_val):
        raise NotImplementedError()


class SGDOptim(Optimizer):

    def get_train_step(self, loss_val, tvars=None):
        if tvars is None:
            tvars = tf.trainable_variables()
        grads = tf.gradients(loss_val, tvars,
                             name='get_grads')
        if 'grad_norm_clip' in self.hyperparams:
            grads, _ = tf.clip_by_global_norm(
                grads, self.hyperparams['grad_norm_clip'])
        optimizer = tf.train.GradientDescentOptimizer(
            self.hyperparams['learning_rate'])
        return optimizer.apply_gradients(zip(grads, tvars))


class AdamOptim(Optimizer):

    def get_train_step(self, loss_val, tvars=None):
        if tvars is None:
            tvars = tf.trainable_variables()
        print(loss_val)
        optimizer = tf.train.AdamOptimizer(
            self.hyperparams['learning_rate'])
        # grads_and_vars = optimizer.compute_gradients(loss_val,
        #                                              var_list=tvars),
        # print(grads_and_vars[0])
        # return optimizer.apply_gradients(grads_and_vars[0])
        return optimizer.minimize(loss_val, var_list=tvars)


class RMSPropOptim(Optimizer):

    def get_train_step(self, loss_val):
        tvars = tf.trainable_variables()
        optimizer = tf.train.RMSPropOptimizer(
            self.hyperparams['learning_rate'],
            decay=self.hyperparams['grad_decay'],
            epsilon=self.hyperparams['grad_epsilon'],
            name='rmsprop_optim')
        grads_and_vars = optimizer.compute_gradients(loss_val, tvars)
        if 'grad_norm_clip' in self.hyperparams:
            norm = self.hyperparams['grad_norm_clip']
            grads_and_vars = [(tf.clip_by_norm(g, norm), v)
                              for (g, v) in grads_and_vars]
        elif 'grad_val_clip' in self.hyperparams:
            val = self.hyperparams['grad_val_clip']
            grads_and_vars = [(tf.clip_by_value(g, -val, val), v)
                              for (g, v) in grads_and_vars]
        return optimizer.apply_gradients(grads_and_vars)


# class RMSPropOptim(Optimizer):

#     def get_train_step(self, loss, tvars=None):
#         if tvars is None:
#             tvars = tf.trainable_variables()
#         optimizer = tf.train.RMSPropOptimizer(
#             self.hyperparams['learning_rate'],
#             decay=self.hyperparams['grad_decay'],
#             epsilon=self.hyperparams['grad_epsilon'],
#             name='rmsprop_optim')
#         return optimizer.minimize(loss, var_list=tvars)


class RMSPropVarOptim(Optimizer):

    def get_train_step(self, loss, learning_rate, tvars=None):
        if tvars is None:
            tvars = tf.trainable_variables()
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=self.hyperparams['grad_decay'],
            epsilon=self.hyperparams['grad_epsilon'],
            name='rmsprop_optim')
        return optimizer.minimize(loss, var_list=tvars)
