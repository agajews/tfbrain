import tensorflow as tf
from tfbrain.helpers import get_supp_train_feed_dict, \
    get_supp_test_feed_dict, iterate_minibatches, \
    get_output


class Trainer(object):

    def __init__(self, hyperparams, ModelClass, LossClass, OptimClass):
        '''hyperparams: a dictionary of hyperparameters
        ModelClass: a child class (not instance) of Model
        loss: a child class of Loss'''
        self.hyperparams = hyperparams
        self.model = ModelClass(hyperparams)
        self.loss = LossClass(hyperparams)
        self.optim = OptimClass(hyperparams)

    def build_model(self):
        '''Builds the Trainer's model,
        generates a TF variable for the
        model's output'''
        self.model.build_net()
        self.y_hat = get_output(self.model.net)
        # self.y_hat = self.model.net.get_output(self.model.net.incoming[0].get_output(None))

    def setup_vars(self):
        self.input_vars = self.model.input_vars
        self.y = tf.placeholder(tf.float32,
                                shape=self.model.net.output_shape)

    def build_loss(self):
        self.loss_val = self.loss.get_loss(self.y_hat, self.y)

    def build_optim(self):
        self.train_step = self.optim.get_train_step(self.loss_val)

    def build_stats(self):
        correct_prediction = tf.equal(tf.argmax(self.y_hat, 1),
                                      tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def create_x_feed_dict(self, batch):
        feed_dict = {}
        for name in batch:
            if not name == 'y':
                feed_dict[self.input_vars[name]] = batch[name]

        return feed_dict

    def create_y_feed_dict(self, y):
        feed_dict = {}
        feed_dict[self.y] = y
        return feed_dict

    def create_supp_train_feed_dict(self):
        supp_feed_dict = get_supp_train_feed_dict(self.model.net)
        return supp_feed_dict

    def create_supp_test_feed_dict(self):
        supp_feed_dict = get_supp_test_feed_dict(self.model.net)
        return supp_feed_dict

    def perform_update(self, batch):
        feed_dict = self.create_x_feed_dict(batch)
        feed_dict.update(self.create_y_feed_dict(batch['y']))
        feed_dict.update(self.create_supp_train_feed_dict())
        self.train_step.run(feed_dict=feed_dict)

    def compute_preds(self, xs):
        feed_dict = self.create_x_feed_dict(xs)
        feed_dict.update(self.create_supp_test_feed_dict())
        preds = self.y_hat.eval(feed_dict=feed_dict)
        return preds

    def compute_accuracy(self, xs, y):
        feed_dict = self.create_x_feed_dict(xs)
        feed_dict.update(self.create_y_feed_dict(y))
        feed_dict.update(self.create_supp_test_feed_dict())
        accuracy = self.accuracy.eval(feed_dict=feed_dict,
                                      session=self.sess)
        return accuracy

    def compute_train_stats(self, batch):
        '''train_xs and train_y are lists of np arrays, not dicts,
        so xs_order is the list of names that belong to each of
        the '''
        return self.compute_accuracy(batch, batch['y'])

    def compute_val_stats(self, val_xs, val_y):
        return self.compute_accuracy(val_xs, val_y)

    def display_update(self, update, epoch,
                      train_stats, val_stats):
        display = ''
        display += 'Step: %d' % update
        display += ', Epoch: %d' % epoch
        if train_stats is not None:
            display += ', Train acc: %f' % train_stats
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
        return iterate_minibatches(inputs, batch_size=self.hyperparams['batch_size'])

    def train(self, train_xs, train_y, val_xs, val_y,
              display_interval=100, cmp_val_stats=False):
        '''train_xs: a dictionary of strings -> np arrays
        matching the model's input_vars dictionary
        train_y: a np array of expected outputs
        val_xs: same as train_xs but for validation
        val_y: same as train_y but for validation'''
        self.build_model()
        self.setup_vars()
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
