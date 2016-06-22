import tfbrain as tb

from tasks.mnist import load_data


class MnistMergeModel(tb.Model):

    def build_net(self):
        i_image_1 = tb.ly.InputLayer(shape=(None, 784))
        net_1 = tb.ly.ReshapeLayer(i_image_1, shape=(None, 28, 28, 1))
        net_1 = tb.ly.Conv2DLayer(net_1, (5, 5), 32)
        net_1 = tb.ly.MaxPool2DLayer(net_1, (2, 2), inner_strides=(2, 2))
        net_1 = tb.ly.Conv2DLayer(net_1, (5, 5), 64)
        net_1 = tb.ly.MaxPool2DLayer(net_1, (2, 2), inner_strides=(2, 2))
        net_1 = tb.ly.FlattenLayer(net_1)
        i_image_2 = tb.ly.InputLayer(shape=(None, 784))
        net_2 = tb.ly.ReshapeLayer(i_image_2, shape=(None, 28, 28, 1))
        net_2 = tb.ly.Conv2DLayer(net_2, (5, 5), 32)
        net_2 = tb.ly.MaxPool2DLayer(net_2, (2, 2), inner_strides=(2, 2))
        net_2 = tb.ly.Conv2DLayer(net_2, (5, 5), 64)
        net_2 = tb.ly.MaxPool2DLayer(net_2, (2, 2), inner_strides=(2, 2))
        net_2 = tb.ly.FlattenLayer(net_2)
        net = tb.ly.MergeLayer([net_1, net_2], axis=1)
        net = tb.ly.FullyConnectedLayer(net, 1024)
        net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net, 10, nonlin=tb.nonlin.softmax)
        self.net = net
        self.input_vars = {'image_1': i_image_1.placeholder,
                           'image_2': i_image_2.placeholder}


def train_merge():
    hyperparams = {'batch_size': 50,
                   'learning_rate': 1e-4,
                   'num_updates': 20000,
                   'grad_norm_clip': 5}
    model = MnistMergeModel(hyperparams)
    loss = tb.Crossentropy(hyperparams)
    acc = tb.CatAcc(hyperparams)
    evaluator = tb.Evaluator(hyperparams, loss, acc)
    optim = tb.AdamOptim(hyperparams)
    trainer = tb.Trainer(model, hyperparams, loss, optim, evaluator)

    mnist = load_data()

    train_xs = {'image_1': mnist['train']['images'],
                'image_2': mnist['train']['images']}
    train_y = mnist['train']['labels']
    val_xs = {'image_1': mnist['test']['images'],
              'image_2': mnist['test']['images']}
    val_y = mnist['test']['labels']

    trainer.train(train_xs, train_y,
                  val_xs, val_y)
    evaluator.eval(model, val_xs, val_y)


if __name__ == '__main__':
    train_merge()
