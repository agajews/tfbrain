class Model(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def build_net(self):
        raise NotImplementedError()
