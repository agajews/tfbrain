from tfbrain.helpers import get_output, \
    create_x_feed_dict, create_supp_test_feed_dict


class Model(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def build_net(self):
        raise NotImplementedError()

    def setup_net(self):
        self.build_net()
        self.y_hat = get_output(self.net)

    def compute_preds(self, xs):
        feed_dict = create_x_feed_dict(self.input_vars, xs)
        feed_dict.update(create_supp_test_feed_dict(self))
        preds = self.y_hat.eval(feed_dict=feed_dict)
        return preds
