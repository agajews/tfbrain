import tensorflow as tf

from tfbrain.helpers import create_x_feed_dict, \
    create_y_feed_dict, create_supp_train_feed_dict, \
    create_minibatch_iterator, avg_over_batches, \
    get_all_params_values
from tfbrain.displays import Display


class Trainer(object):

    def __init__(self,
                 model,
                 hyperparams,
                 LossClass,
                 AccuracyClass,
                 OptimClass,
                 DisplayClass=Display):
        '''hyperparams: a dictionary of hyperparameters
        ModelClass: a child class (not instance) of Model
        loss: a child class of Loss'''
        self.hyperparams = hyperparams
        self.model = model
        self.loss = LossClass(hyperparams)
        self.accuracy = AccuracyClass(hyperparams)
        self.optim = OptimClass(hyperparams)
        self.display = DisplayClass(hyperparams)

    def build_vars(self, target_dtype):
        self.y = tf.placeholder(target_dtype,
                                shape=self.model.net.output_shape)

    def build_loss(self):
        self.loss.build(self.model.y_hat, self.y)

    def build_optim(self):
        self.train_step = self.optim.get_train_step(self.loss.loss)

    def build_stats(self):
        self.accuracy.build(self.model.y_hat, self.y)

    def perform_update(self, batch):
        feed_dict = create_x_feed_dict(self.model.input_vars, batch)
        feed_dict.update(create_y_feed_dict(self.y, batch['y']))
        feed_dict.update(create_supp_train_feed_dict(self.model))
        self.train_step.run(feed_dict=feed_dict)

    def compute_train_stats(self, batch):
        return {'acc': self.accuracy.compute(
                    self.model, batch, self.y, batch['y']),
                'loss': self.loss.compute(
                    self.model, batch, self.y, batch['y'])}

    def compute_val_stats(self,
                          val_xs,
                          val_y):

        minibatches = create_minibatch_iterator(
            val_xs, val_y, self.model.test_batch_preprocessor,
            batch_size=self.hyperparams['batch_size'])

        def fn(batch):
            return self.accuracy.compute(
                self.model, batch, self.y, batch['y'])

        acc = avg_over_batches(minibatches, fn)
        return {'acc': acc}

    def init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def train(self, train_xs, train_y, val_xs, val_y,
              target_dtype=tf.float32,
              display_interval=100, cmp_val_stats=False):
        '''train_xs: a dictionary of strings -> np arrays
        matching the model's input_vars dictionary
        train_y: a np array of expected outputs
        val_xs: same as train_xs but for validation
        val_y: same as train_y but for validation'''
        self.model.setup_net()
        self.build_vars(target_dtype)
        self.build_loss()
        self.build_optim()
        self.build_stats()
        self.init_session()

        def recreate_batches():
            return create_minibatch_iterator(
                train_xs, train_y, self.model.train_batch_preprocessor,
                batch_size=self.hyperparams['batch_size'])

        minibatches = recreate_batches()
        epoch = 0
        for update in range(self.hyperparams['num_updates']):
            try:
                batch = next(minibatches)
            except StopIteration:
                minibatches = recreate_batches()
                batch = next(minibatches)
                epoch += 1
            self.perform_update(batch)
            if update % display_interval == 0:
                train_stats = self.compute_train_stats(batch)
                if cmp_val_stats:
                    val_stats = self.compute_val_stats(
                        val_xs, val_y)
                else:
                    val_stats = None
                self.display.update(
                    update, epoch, train_stats, val_stats,
                    model=self.model)

    def eval(self,
             val_xs,
             val_y):
        val_stats = self.compute_val_stats(val_xs, val_y)
        self.display.val_stats(val_stats,
                               model=self.model)
