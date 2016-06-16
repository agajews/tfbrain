from tasks.char_rnn import gen_sequence


class Display(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def update(self, update, epoch,
               train_stats, val_stats,
               **kwargs):
        display = ''
        display += 'Step: %d' % update
        display += ', Epoch: %d' % epoch
        if train_stats is not None:
            display += ', Train loss: %f' % train_stats['loss']
            display += ', Train acc: %f' % train_stats['acc']
        if val_stats is not None:
            display += ', Val acc: %f' % val_stats['acc']

        print(display)

    def val_stats(self, val_stats, **kwargs):
        print('Val acc: %f' % val_stats['acc'])


class PerplexityDisplay(Display):

    def update(self, update, epoch,
               train_stats, val_stats,
               **kwargs):
        display = ''
        display += 'Step: %d' % update
        display += ', Epoch: %d' % epoch
        if train_stats is not None:
            display += ', Train loss: %f' % train_stats['loss']
            display += ', Train perplexity: %f' % train_stats['acc']
        if val_stats is not None:
            display += ', Val perplexity: %f' % val_stats['acc']

        print(display)

    def val_stats(self, val_stats, **kwargs):
        print('Val perplexity: %f' % val_stats['acc'])


class SeqGenDisplay(Display):

    def update(self, update, epoch,
               train_stats, val_stats,
               model=None, **kwargs):
        PerplexityDisplay.update(self, update, epoch,
                                 train_stats, val_stats,
                                 **kwargs)
        random_snippet = gen_sequence(
            model, self.hyperparams['seqlength'],
            self.hyperparams['char_to_index'],
            self.hyperparams['index_to_char'])

        print('-----\n%s\n-----' % random_snippet)
