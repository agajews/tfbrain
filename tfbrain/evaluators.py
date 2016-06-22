from tasks.char_rnn import gen_sequence

from tfbrain.helpers import create_minibatch_iterator, \
    avg_over_batches


class Evaluator(object):

    def __init__(self, hyperparams, loss, acc, **kwargs):
        self.hyperparams = hyperparams
        self.loss = loss
        self.accuracy = acc
        self.kwargs = kwargs

    def build(self, model, y_var, train_mask):
        self.y_var = y_var
        self.loss.build(model.y_hat, y_var, train_mask)
        if self.accuracy is not None:
            self.accuracy.build(model.y_hat, y_var)

    def compute_train_stats(self, model, batch):
        stats = {}
        stats['loss'] = self.loss.compute(
            model, batch, self.y_var, batch['y'])
        if self.accuracy is not None:
            stats['acc'] = self.accuracy.compute(
                model, batch, self.y_var, batch['y'])
        return stats

    def compute_val_stats(self,
                          model,
                          val_xs, val_y):

        if self.accuracy is not None:
            minibatches = create_minibatch_iterator(
                val_xs, val_y, model.test_batch_preprocessor,
                batch_size=self.hyperparams['batch_size'])

            def fn(batch):
                return self.accuracy.compute(
                    model, batch, self.y_var, batch['y'])

            acc = avg_over_batches(minibatches, fn)
            return {'acc': acc}
        else:
            return None

    def build_update_display(self, update, epoch,
                             train_stats, val_stats):
        display = ''
        display += 'Step: %d' % update
        display += ', Epoch: %d' % epoch
        if train_stats is not None:
            display += ', Train loss: %f' % train_stats['loss']
            if 'acc' in train_stats.keys():
                display += ', Train acc: %f' % train_stats['acc']
        if val_stats is not None:
            display += ', Val acc: %f' % val_stats['acc']
        return display

    def display_update(self, model,
                       batch,
                       val_xs, val_y,
                       update, epoch,
                       val_cmp=False):
        train_stats = self.compute_train_stats(model, batch)
        if val_cmp:
            val_stats = self.compute_val_stats(model, val_xs, val_y)
        else:
            val_stats = None
        display = self.build_update_display(
            update, epoch, train_stats, val_stats)
        print(display)

    def build_val_stats_display(self, val_stats):
        display = 'Val acc: %f' % val_stats['acc']
        return display

    def eval(self, model, val_xs, val_y):
        val_stats = self.compute_val_stats(
            model, val_xs, val_y)
        display = self.build_val_stats_display(val_stats)
        print(display)


class PerplexityEvaluator(Evaluator):

    def build_update_display(self, update, epoch,
                             train_stats, val_stats):
        display = ''
        display += 'Step: %d' % update
        display += ', Epoch: %d' % epoch
        if train_stats is not None:
            display += ', Train loss: %f' % train_stats['loss']
            display += ', Train perplexity: %f' % train_stats['acc']
        if val_stats is not None:
            display += ', Val perplexity: %f' % val_stats['acc']
        return display

    def build_val_stats_display(self, val_stats):
        display = 'Val perplexity: %f' % val_stats['acc']
        return display


class SeqGenEvaluator(PerplexityEvaluator):

    def display_update(self, model,
                       batch,
                       val_xs, val_y,
                       update, epoch,
                       val_cmp=False):

        Evaluator.display_update(self, model,
                                 batch,
                                 val_xs, val_y,
                                 update, epoch,
                                 val_cmp)

        random_snippet = gen_sequence(
            model, self.hyperparams['seqlength'],
            self.hyperparams['char_to_index'],
            self.hyperparams['index_to_char'],
            seed=self.kwargs['seed'])

        print('-----\n%s\n-----' % random_snippet)
