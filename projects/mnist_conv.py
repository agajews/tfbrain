from tfbrain import nonlin
from tfbrain.layers import InputLayer, FullyConnectedLayer, \
    ReshapeLayer, Conv2DLayer, MaxPool2DLayer, FlattenLayer, \
    DropoutLayer
from tfbrain.trainers import Trainer
from tfbrain.loss import CrossentropyLoss
from tfbrain.optim import AdamOptim
from tfbrain.models import Model
from tfbrain.acc import CatAcc

from datasets.mnist import load_data


class MnistConvModel(Model):

    def build_net(self):
        i_image = InputLayer(shape=(None, 784))
        net = ReshapeLayer(i_image, shape=(None, 28, 28, 1))
        net = Conv2DLayer(net, (5, 5), 32)
        net = MaxPool2DLayer(net, (2, 2), inner_strides=(2, 2))
        net = Conv2DLayer(net, (5, 5), 64)
        net = MaxPool2DLayer(net, (2, 2), inner_strides=(2, 2))
        net = FlattenLayer(net)
        net = FullyConnectedLayer(net, 1024)
        net = DropoutLayer(net, 0.5)
        net = FullyConnectedLayer(net, 10, nonlin=nonlin.softmax)
        self.net = net
        self.input_vars = {'image': i_image.placeholder}


def train_conv():
    hyperparams = {'batch_size': 50,
                   'learning_rate': 1e-4,
                   'num_updates': 20000}
    model = MnistConvModel(hyperparams)
    trainer = Trainer(model, hyperparams, CrossentropyLoss, CatAcc, AdamOptim)

    mnist = load_data()

    train_xs = {'image': mnist['train']['images']}
    train_y = mnist['train']['labels']
    val_xs = {'image': mnist['test']['images']}
    val_y = mnist['test']['labels']

    trainer.train(train_xs, train_y,
                  val_xs, val_y)
    trainer.eval(val_xs, val_y)


if __name__ == '__main__':
    train_conv()
