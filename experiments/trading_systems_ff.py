import tfbrain as tb

import tensorflow as tf

import numpy as np


class TradingSystemsModel(tb.Model):

    def build_net(self):
        ticks = tb.ly.InputLayer(shape=(None, 100, 2),
                                 name='ticks')
        ticks = tb.ly.FlattenLayer(ticks)
        net = tb.ly.FullyConnectedLayer(ticks, 1024)
        net = tb.ly.FullyConnectedLayer(net, 1024)
        net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net, 2,
                                        nonlin=tb.nonlin.softmax)
        self.net = net


def train():
    hyperparams = {'batch_size': 50,
                   'learning_rate': 0.0001,
                   'grad_decay': 0.95,
                   'grad_epsilon': 0.01,
                   'num_updates': 20000,
                   'grad_norm_clip': 5}
    with tf.device('/cpu:0'):
        model = TradingSystemsModel(hyperparams)
    loss = tb.Crossentropy(hyperparams)
    acc = tb.CatAcc(hyperparams)
    evaluator = tb.Evaluator(hyperparams, loss, acc)
    optim = tb.RMSPropOptim(hyperparams)
    trainer = tb.Trainer(model, hyperparams, loss, optim, evaluator)

    split = 90000
    data = np.load('data/trading-systems.npz')
    print(data['ticks'].shape)
    train_xs = {'ticks': data['ticks'][:split]}
    train_y = data['targets'][:split]

    val_xs = {'ticks': data['ticks'][split:]}
    val_y = data['targets'][split:]

    with tf.device('/cpu:0'):
        trainer.train(train_xs, train_y,
                      val_xs, val_y,
                      val_cmp=True)
    evaluator.eval(model, val_xs, val_y)


if __name__ == '__main__':
    train()
