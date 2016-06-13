from tfbrain import nonlin
from tfbrain.layers import InputLayer, FullyConnectedLayer, \
    BasicRNNLayer, SliceLayer, DropoutLayer, EmbeddingLayer
from tfbrain.trainers import UnsupervisedTrainer
from tfbrain.loss import LogProbLoss
from tfbrain.optim import AdamOptim
from tfbrain.models import Model
from tfbrain.acc import Perplexity

from datasets import indices_to_seq_data

from tensorflow.models.rnn.ptb import reader


class MnistConvModel(Model):

    def build_net(self):
        vocab_size = self.hyperparams['vocab_size']
        i_text = InputLayer(shape=(None, None))
        net = EmbeddingLayer(i_text, 100, vocab_size)
        net = BasicRNNLayer(net, 100)
        net = SliceLayer(net, axis=-1)
        net = FullyConnectedLayer(net, 1024)
        net = DropoutLayer(net, 0.5)
        net = FullyConnectedLayer(net, vocab_size, nonlin=nonlin.softmax)
        self.net = net
        self.input_vars = {'text': i_text.placeholder}


def train_brnn():
    vocab_size = 10000
    seqlength = 2
    hyperparams = {'batch_size': 50,
                   'learning_rate': 1e-4,
                   'num_updates': 20000,
                   'vocab_size': vocab_size}
    model = MnistConvModel(hyperparams)
    trainer = UnsupervisedTrainer(
        model, hyperparams, LogProbLoss, Perplexity, AdamOptim)

    raw_data = reader.ptb_raw_data('data/ptb')
    train_data, val_data, test_data, _ = raw_data
    train_data = indices_to_seq_data(train_data, seqlength)
    val_data = indices_to_seq_data(val_data, seqlength)

    trainer.train(train_data, val_data)
    trainer.eval(val_data)


if __name__ == '__main__':
    train_brnn()
