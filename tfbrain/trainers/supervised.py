import tensorflow as tf
from tfbrain.helpers import create_x_feed_dict, \
    create_y_feed_dict, create_supp_train_feed_dict, \
    create_supp_test_feed_dict, iterate_minibatches


class Trainer(object):

    def __init__(self,
                 model,
                 hyperparams,
                 LossClass,
                 AccuracyClass,
                 OptimClass):
        '''hyperparams: a dictionary of hyperparameters
        ModelClass: a child class (not instance) of Model
        loss: a child class of Loss'''
        self.hyperparams = hyperparams
        self.model = model
        self.loss_obj = LossClass(hyperparams)
        self.accuracy_obj = AccuracyClass(hyperparams)
        self.optim = OptimClass(hyperparams)

    def setup_vars(self, target_dtype):
        self.y = tf.placeholder(target_dtype,
                                shape=self.model.net.output_shape)

    def build_loss(self):
        self.loss = self.loss_obj.build(self.model.y_hat, self.y)

    def build_optim(self):
        self.train_step = self.optim.get_train_step(self.loss)

    def build_stats(self):
        self.accuracy = self.accuracy_obj.build(self.model.y_hat, self.y)

    def perform_update(self, batch):
        feed_dict = create_x_feed_dict(self.model.input_vars, batch)
        feed_dict.update(create_y_feed_dict(self.y, batch['y']))
        feed_dict.update(create_supp_train_feed_dict(self.model))
        self.train_step.run(feed_dict=feed_dict)

    def compute_accuracy(self, xs, y):
        feed_dict = create_x_feed_dict(self.model.input_vars, xs)
        feed_dict.update(create_y_feed_dict(self.y, y))
        feed_dict.update(create_supp_test_feed_dict(self.model))
        accuracy = self.accuracy.eval(feed_dict=feed_dict,
                                      session=self.sess)
        return accuracy

    def compute_loss(self, xs, y):
        feed_dict = create_x_feed_dict(self.model.input_vars, xs)
        feed_dict.update(create_y_feed_dict(self.y, y))
        feed_dict.update(create_supp_test_feed_dict(self.model))
        accuracy = self.loss.eval(feed_dict=feed_dict,
                                  session=self.sess)
        return accuracy

    def compute_train_stats(self, batch):
        return {'acc': self.compute_accuracy(batch, batch['y']),
                'loss': self.compute_loss(batch, batch['y'])}

    def compute_val_stats(self, val_xs, val_y):
        return self.compute_accuracy(val_xs, val_y)

    def display_update(self, update, epoch,
                       train_stats, val_stats):
        display = ''
        display += 'Step: %d' % update
        display += ', Epoch: %d' % epoch
        if train_stats is not None:
            display += ', Train loss: %f' % train_stats['loss']
            display += ', Train acc: %f' % train_stats['acc']
        if val_stats is not None:
            display += ', Val acc: %f' % val_stats

        print(display)

    def init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def display_val_stats(self, val_stats):
        print('Val acc: %f' % val_stats)

    def create_minibatch_iterator(self, train_xs, train_y):
        inputs = {}
        inputs.update(train_xs)
        inputs['y'] = train_y
        return iterate_minibatches(
            inputs, batch_size=self.hyperparams['batch_size'])

    def train(self, train_xs, train_y, val_xs, val_y,
              target_dtype=tf.float32,
              display_interval=100, cmp_val_stats=False):
        '''train_xs: a dictionary of strings -> np arrays
        matching the model's input_vars dictionary
        train_y: a np array of expected outputs
        val_xs: same as train_xs but for validation
        val_y: same as train_y but for validation'''
        self.model.setup_net()
        self.setup_vars(target_dtype)
        self.build_loss()
        self.build_optim()
        self.build_stats()
        self.init_session()

        minibatches = self.create_minibatch_iterator(train_xs, train_y)
        epoch = 0
        for update in range(self.hyperparams['num_updates']):
            try:
                batch = next(minibatches)
            except StopIteration:
                minibatches = self.create_minibatch_iterator(
                    train_xs, train_y)
                batch = next(minibatches)
                epoch += 1
            self.perform_update(batch)
            if update % display_interval == 0:
                train_stats = self.compute_train_stats(batch)
                if cmp_val_stats:
                    val_stats = self.compute_val_stats(val_xs, val_y)
                else:
                    val_stats = None
                self.display_update(update, epoch, train_stats, val_stats)

    def eval(self, val_xs, val_y):
        val_stats = self.compute_val_stats(val_xs, val_y)
        self.display_val_stats(val_stats)
