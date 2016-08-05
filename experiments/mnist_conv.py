import tfbrain as tb

from tasks.mnist import load_data


class MnistConvModel(tb.Model):

    def build_net(self):
        i_image = tb.ly.InputLayer(shape=(None, 784),
                                   name='image')
        net = tb.ly.ReshapeLayer(i_image, shape=(None, 28, 28, 1))
        net = tb.ly.Conv2DLayer(net, (5, 5), 32)
        net = tb.ly.MaxPool2DLayer(net, (2, 2), inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (5, 5), 64)
        net = tb.ly.MaxPool2DLayer(net, (2, 2), inner_strides=(2, 2))
        net = tb.ly.FlattenLayer(net)
        net = tb.ly.FullyConnectedLayer(net, 1024)
        net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net, 10, nonlin=tb.nonlin.softmax)
        self.net = net


def train_conv():
    hyperparams = {'batch_size': 50,
                   'learning_rate': 1e-4,
                   'num_updates': 20000,
                   'grad_norm_clip': 5}
    model = MnistConvModel(hyperparams)
    loss = tb.Crossentropy(hyperparams)
    acc = tb.CatAcc(hyperparams)
    evaluator = tb.Evaluator(hyperparams, loss, acc)
    optim = tb.AdamOptim(hyperparams)
    trainer = tb.Trainer(model, hyperparams, loss, optim, evaluator)

    mnist = load_data()

    train_xs = {'image': mnist['train']['images']}
    train_y = mnist['train']['labels']
    val_xs = {'image': mnist['test']['images']}
    val_y = mnist['test']['labels']

    trainer.build()
    trainer.train(train_xs, train_y,
                  val_xs, val_y,
                  build=False)
    evaluator.eval(model, val_xs, val_y)


if __name__ == '__main__':
    train_conv()
