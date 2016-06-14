from tfbrain import nonlin
from tfbrain.layers import InputLayer, FullyConnectedLayer, \
    LSTMLayer, SeqSliceLayer, DropoutLayer, EmbeddingLayer
from tfbrain.trainers import UnhotCatTrainer
from tfbrain.loss import Crossentropy
from tfbrain.optim import AdamOptim
from tfbrain.models import Model
from tfbrain.acc import Perplexity

from datasets import indices_to_seq_data

import tensorflow as tf
from tensorflow.models.rnn.ptb import reader


class PTBBasic(Model):

    def build_net(self):
        vocab_size = self.hyperparams['vocab_size']
        i_text = InputLayer(shape=(None, None),
                            dtype=tf.int32)
        net = EmbeddingLayer(i_text, 200, vocab_size)
        net = LSTMLayer(net, 200)
        net = DropoutLayer(net, 0.5)
        net = LSTMLayer(net, 200)
        net = DropoutLayer(net, 0.5)
        net = SeqSliceLayer(net, col=-1)
        net = FullyConnectedLayer(net, 1024)
        net = DropoutLayer(net, 0.5)
        net = FullyConnectedLayer(net, vocab_size, nonlin=nonlin.softmax)
        self.net = net
        self.input_vars = {'text': i_text.placeholder}


def train_brnn():
    vocab_size = 10000
    seqlength = 20
    hyperparams = {'batch_size': 50,
                   'learning_rate': 1e-4,
                   'num_updates': 200,
                   'vocab_size': vocab_size}
    model = PTBBasic(hyperparams)
    trainer = UnhotCatTrainer(
        model, hyperparams, Crossentropy, Perplexity, AdamOptim)

    raw_data = reader.ptb_raw_data('data/ptb')
    train_data, val_data, test_data, _ = raw_data
    train_data = indices_to_seq_data(train_data, seqlength)
    val_data = indices_to_seq_data(val_data, seqlength)

    train_xs = {'text': train_data['text']}
    train_y = train_data['targets']
    val_xs = {'text': val_data['text']}
    val_y = val_data['targets']

    trainer.train(train_xs, train_y,
                  val_xs, val_y,
                  num_cats=vocab_size)
    trainer.eval(val_xs, val_y, vocab_size)


if __name__ == '__main__':
    train_brnn()
