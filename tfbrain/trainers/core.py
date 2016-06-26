import tensorflow as tf

from tfbrain.helpers import create_x_feed_dict, \
    create_y_feed_dict, create_supp_train_feed_dict, \
    create_minibatch_iterator


class Trainer(object):

    def __init__(self,
                 model,
                 hyperparams,
                 loss,
                 optim,
                 evaluator):
        '''hyperparams: a dictionary of hyperparameters
        ModelClass: a child class (not instance) of Model
        loss: a child class of Loss'''
        self.hyperparams = hyperparams
        self.model = model
        self.optim = optim
        self.evaluator = evaluator

    def build_vars(self, train_mask_shape, target_dtype):
        if train_mask_shape is not None:
            self.train_mask = tf.placeholder(shape=train_mask_shape,
                                             dtype=tf.float32)
            self.model.input_vars['mask'] = self.train_mask
        else:
            self.train_mask = None
        self.y = tf.placeholder(target_dtype,
                                shape=self.model.get_net().get_output_shape())

    def build_eval(self):
        self.evaluator.build(self.model, self.y, self.train_mask)

    def build_optim(self):
        self.train_step = self.optim.get_train_step(self.evaluator.loss.loss)

    def perform_update(self, batch):
        feed_dict = create_x_feed_dict(self.model.input_vars, batch)
        feed_dict.update(create_y_feed_dict(self.y, batch['y']))
        feed_dict.update(create_supp_train_feed_dict(self.model))
        self.train_step.run(feed_dict=feed_dict)

    def init_session(self):
        self.sess = tf.InteractiveSession()
        print('Initializing variables ...')
        self.sess.run(tf.initialize_all_variables())
        print('Initialized variables ...')

    def build(self, train_mask_shape=None, target_dtype=tf.float32):
        print('Building trainer ...')
        self.model.setup_net()
        self.build_vars(train_mask_shape, target_dtype)
        self.build_eval()
        self.build_optim()
        self.init_session()

    def train(self, train_xs, train_y, val_xs, val_y,
              train_mask=None,
              target_dtype=tf.float32,
              display_interval=100, val_cmp=False,
              build=True, num_updates=None):
        '''train_xs: a dictionary of strings -> np arrays
        matching the model's input_vars dictionary
        train_y: a np array of expected outputs
        val_xs: same as train_xs but for validation
        val_y: same as train_y but for validation'''

        if num_updates is None:
            num_updates = self.hyperparams['num_updates']

        if build:
            if train_mask is not None:
                train_mask_shape = train_mask.shape
            else:
                train_mask_shape = None
            self.build(train_mask_shape, target_dtype)

        def recreate_batches():
            return create_minibatch_iterator(
                train_xs, train_y,
                self.model.train_batch_preprocessor,
                batch_size=self.hyperparams['batch_size'],
                train_mask=train_mask)

        minibatches = recreate_batches()
        epoch = 0
        for update in range(num_updates):
            try:
                batch = next(minibatches)
            except StopIteration:
                minibatches = recreate_batches()
                batch = next(minibatches)
                epoch += 1
            self.perform_update(batch)
            if display_interval is not None and \
                    update % display_interval == 0:
                self.evaluator.display_update(
                    self.model, batch, val_xs, val_y, update, epoch,
                    val_cmp)
