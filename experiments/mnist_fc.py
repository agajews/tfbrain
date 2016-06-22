import tfbrain as tb

from tasks.mnist import load_data


class MnistFCModel(tb.Model):

    def build_net(self):
        i_image = tb.ly.InputLayer(shape=(None, 784))
        net = tb.ly.FullyConnectedLayer(i_image, 50, nonlin=tb.nonlin.tanh)
        net = tb.ly.FullyConnectedLayer(net, 10, nonlin=tb.nonlin.softmax)
        self.net = net
        self.input_vars = {'image': i_image.placeholder}


def train_fc():
    hyperparams = {'batch_size': 50,
                   'learning_rate': 0.5,
                   'num_updates': 2000,
                   'grad_norm_clip': 5}
    model = MnistFCModel(hyperparams)
    loss = tb.Crossentropy(hyperparams)
    acc = tb.CatAcc(hyperparams)
    evaluator = tb.Evaluator(hyperparams, loss, acc)
    optim = tb.SGDOptim(hyperparams)
    trainer = tb.Trainer(model, hyperparams, loss, optim, evaluator)

    mnist = load_data()

    train_xs = {'image': mnist['train']['images']}
    train_y = mnist['train']['labels']
    val_xs = {'image': mnist['test']['images']}
    val_y = mnist['test']['labels']

    trainer.train(train_xs, train_y,
                  val_xs, val_y)
    evaluator.eval(model, val_xs, val_y)


if __name__ == '__main__':
    train_fc()
