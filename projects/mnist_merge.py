from tfbrain import nonlin
from tfbrain.layers import InputLayer, FullyConnectedLayer, \
    ReshapeLayer, Conv2DLayer, MaxPool2DLayer, FlattenLayer, \
    DropoutLayer, MergeLayer
from tfbrain.trainers import Trainer
from tfbrain.loss import CrossentropyLoss
from tfbrain.optim import AdamOptim
from tfbrain.models import Model
from tfbrain.acc import CatAcc

from datasets.mnist import load_data


class MnistMergeModel(Model):

    def build_net(self):
        i_image_1 = InputLayer(shape=(None, 784))
        net_1 = ReshapeLayer(i_image_1, shape=(None, 28, 28, 1))
        net_1 = Conv2DLayer(net_1, (5, 5), 32)
        net_1 = MaxPool2DLayer(net_1, (2, 2), inner_strides=(2, 2))
        net_1 = Conv2DLayer(net_1, (5, 5), 64)
        net_1 = MaxPool2DLayer(net_1, (2, 2), inner_strides=(2, 2))
        net_1 = FlattenLayer(net_1)
        i_image_2 = InputLayer(shape=(None, 784))
        net_2 = ReshapeLayer(i_image_2, shape=(None, 28, 28, 1))
        net_2 = Conv2DLayer(net_2, (5, 5), 32)
        net_2 = MaxPool2DLayer(net_2, (2, 2), inner_strides=(2, 2))
        net_2 = Conv2DLayer(net_2, (5, 5), 64)
        net_2 = MaxPool2DLayer(net_2, (2, 2), inner_strides=(2, 2))
        net_2 = FlattenLayer(net_2)
        net = MergeLayer([net_1, net_2], axis=1)
        net = FullyConnectedLayer(net, 1024)
        net = DropoutLayer(net, 0.5)
        net = FullyConnectedLayer(net, 10, nonlin=nonlin.softmax)
        self.net = net
        self.input_vars = {'image_1': i_image_1.placeholder,
                           'image_2': i_image_2.placeholder}


def train_merge():
    hyperparams = {'batch_size': 50,
                   'learning_rate': 1e-4,
                   'num_updates': 20000}
    model = MnistMergeModel(hyperparams)
    trainer = Trainer(model, hyperparams, CrossentropyLoss, CatAcc, AdamOptim)

    mnist = load_data()

    train_xs = {'image_1': mnist['train']['images'],
                'image_2': mnist['train']['images']}
    train_y = mnist['train']['labels']
    val_xs = {'image_1': mnist['test']['images'],
              'image_2': mnist['test']['images']}
    val_y = mnist['test']['labels']

    trainer.train(train_xs, train_y,
                  val_xs, val_y)
    trainer.eval(val_xs, val_y)


if __name__ == '__main__':
    train_merge()
