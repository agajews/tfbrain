import tfbrain as tb

from tasks.ptb import load_data

# import tensorflow as tf


class PTBBasic(tb.UnhotXYModel):

    def build_net(self):
        vocab_size = self.hyperparams['vocab_size']
        self.num_cats = vocab_size
        i_text = tb.ly.InputLayer(shape=(None, None, vocab_size))
        # net = tb.ly.EmbeddingLayer(i_text, 200, vocab_size)
        net = i_text
        net = tb.ly.LSTMLayer(net, 200)
        net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.LSTMLayer(net, 200)
        net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.SeqSliceLayer(net, col=-1)
        net = tb.ly.FullyConnectedLayer(net, 1024)
        net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(
            net, vocab_size, nonlin=tb.nonlin.softmax)
        self.net = net
        self.input_vars = {'text': i_text.placeholder}


def train_brnn():
    vocab_size = 10000
    seqlength = 20
    hyperparams = {'batch_size': 20,
                   'learning_rate': 1e-4,
                   'num_updates': 5000,
                   'grad_norm_clip': 5,
                   'vocab_size': vocab_size}
    model = PTBBasic(hyperparams)
    trainer = tb.Trainer(
        model, hyperparams, tb.Crossentropy, tb.Perplexity, tb.AdamOptim,
        DisplayClass=tb.PerplexityDisplay)

    print('Loading data...')
    data = load_data(seqlength)

    train_xs = {'text': data['train']['text']}
    train_y = data['train']['targets']
    val_xs = {'text': data['test']['text']}
    val_y = data['test']['targets']

    print('Training...')

    trainer.train(train_xs, train_y,
                  val_xs, val_y)
    trainer.eval(val_xs, val_y)


if __name__ == '__main__':
    train_brnn()
