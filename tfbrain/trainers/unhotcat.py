import tensorflow as tf

from tfbrain.helpers import iterate_minibatches
from datasets import labels_to_one_hot
from .supervised import Trainer


class UnhotCatTrainer(Trainer):

    def train(self, train_xs, train_y, val_xs, val_y,
              num_cats, target_dtype=tf.float32,
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

        minibatches = self.create_minibatch_iterator(train_xs, train_y, num_cats)
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

    def create_minibatch_iterator(self, train_xs, train_y, num_cats):
        inputs = {}
        inputs.update(train_xs)
        inputs['y'] = train_y
        minibatches = iterate_minibatches(
            inputs, batch_size=self.hyperparams['batch_size'])

        def one_hot_fn(batch):
            batch['y'] = labels_to_one_hot(batch['y'], num_cats)
            return batch

        return map(one_hot_fn, minibatches)

    def eval(self, val_xs, val_y, num_cats):
        minibatches = self.create_minibatch_iterator(val_xs, val_y, num_cats)
        val_acc = 0
        num_batches = 0
        for batch in minibatches:
            val_acc += self.compute_val_stats(batch, batch['y'])
            num_batches += 1
        val_acc = val_acc / num_batches
        self.display_val_stats(val_acc)
