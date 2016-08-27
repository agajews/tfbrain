import tfbrain as tb

import numpy as np


class CoinbaseModel(tb.Model):

    def build_net(self):
        first_ticks = tb.ly.InputLayer(shape=(None, None, 2),
                                       name='first_ticks')
        first_ticks = tb.ly.LSTMLayer(first_ticks, 512)
        first_ticks = tb.ly.DropoutLayer(first_ticks, 0.1)
        first_ticks = tb.ly.SeqSliceLayer(first_ticks, col=-1)
        first_ticks = tb.ly.FullyConnectedLayer(first_ticks, 512)

        last_ticks = tb.ly.InputLayer(shape=(None, None, 2),
                                      name='last_ticks')
        last_ticks = tb.ly.LSTMLayer(last_ticks, 512)
        last_ticks = tb.ly.DropoutLayer(last_ticks, 0.1)
        last_ticks = tb.ly.SeqSliceLayer(last_ticks, col=-1)
        last_ticks = tb.ly.FullyConnectedLayer(last_ticks, 512)

        features = tb.ly.InputLayer(shape=(None, 3),
                                    name='features')
        features = tb.ly.FullyConnectedLayer(features, 512)

        net = tb.ly.MergeLayer([first_ticks, last_ticks, features])
        net = tb.ly.FullyConnectedLayer(net, 1024)
        net = tb.ly.DropoutLayer(net, 0.1)
        net = tb.ly.FullyConnectedLayer(net, 2,
                                        nonlin=tb.nonlin.softmax)
        self.net = net


def train():
    hyperparams = {'batch_size': 50,
                   'learning_rate': 0.001,
                   'grad_decay': 0.99,
                   'grad_epsilon': 0.01,
                   'num_updates': 20000,
                   'grad_norm_clip': 5}
    model = CoinbaseModel(hyperparams)
    loss = tb.Crossentropy(hyperparams)
    acc = tb.CatAcc(hyperparams)
    evaluator = tb.Evaluator(hyperparams, loss, acc)
    optim = tb.RMSPropOptim(hyperparams)
    trainer = tb.Trainer(model, hyperparams, loss, optim, evaluator)

    split = 10000
    data = np.load('data/coinbase_n1.npz')
    train_xs = {'first_ticks': data['first_ticks'][:split],
                'last_ticks': data['last_ticks'][:split],
                'features': data['features'][:split]}
    train_y = data['targets'][:split]

    val_xs = {'first_ticks': data['first_ticks'][split:],
              'last_ticks': data['last_ticks'][split:],
              'features': data['features'][split:]}
    val_y = data['targets'][split:]

    trainer.train(train_xs, train_y,
                  val_xs, val_y,
                  val_cmp=True)
    evaluator.eval(model, val_xs, val_y)


if __name__ == '__main__':
    train()
