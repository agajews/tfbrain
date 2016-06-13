from tfbrain import nonlin
from tfbrain.layers import InputLayer, FullyConnectedLayer
from tfbrain.trainers import Trainer
from tfbrain.loss import CrossentropyLoss
from tfbrain.optim import SGDOptim
from tfbrain.models import Model
from tfbrain.acc import CatAcc

from datasets.mnist import load_data


class MnistFCModel(Model):

    def build_net(self):
        i_image = InputLayer(shape=(None, 784))
        net = FullyConnectedLayer(i_image, 50, nonlin=nonlin.tanh)
        net = FullyConnectedLayer(net, 10, nonlin=nonlin.softmax)
        self.net = net
        self.input_vars = {'image': i_image.placeholder}


def train_fc():
    hyperparams = {'batch_size': 50,
                   'learning_rate': 0.5,
                   'num_updates': 2000}
    model = MnistFCModel(hyperparams)
    trainer = Trainer(model, hyperparams, CrossentropyLoss, CatAcc, SGDOptim)

    mnist = load_data()

    train_xs = {'image': mnist['train']['images']}
    train_y = mnist['train']['labels']
    val_xs = {'image': mnist['test']['images']}
    val_y = mnist['test']['labels']

    trainer.train(train_xs, train_y,
                  val_xs, val_y)
    trainer.eval(val_xs, val_y)


if __name__ == '__main__':
    train_fc()
