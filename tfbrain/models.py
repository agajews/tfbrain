from tfbrain.helpers import get_output, \
    create_x_feed_dict, create_supp_test_feed_dict

from tasks import labels_to_one_hot


class Model(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def build_net(self):
        raise NotImplementedError()

    def train_batch_preprocessor(self, batch):
        return batch

    def test_batch_preprocessor(self, batch):
        return batch

    def pred_xs_preprocessor(self, xs):
        return xs

    def setup_net(self):
        self.build_net()
        self.y_hat = get_output(self.net)

    def compute_preds(self, xs):
        xs = self.pred_xs_preprocessor(xs)
        feed_dict = create_x_feed_dict(self.input_vars, xs)
        feed_dict.update(create_supp_test_feed_dict(self))
        preds = self.y_hat.eval(feed_dict=feed_dict)
        return preds


class UnhotYModel(Model):

    def train_batch_preprocessor(self, batch):
        batch['y'] = labels_to_one_hot(batch['y'], self.num_cats)
        return batch

    def test_batch_preprocessor(self, batch):
        batch['y'] = labels_to_one_hot(batch['y'], self.num_cats)
        return batch


class UnhotXYModel(Model):

    def train_batch_preprocessor(self, batch):
        for x_name in self.input_vars:
            batch[x_name] = labels_to_one_hot(batch[x_name], self.num_cats)
        batch['y'] = labels_to_one_hot(batch['y'], self.num_cats)
        return batch

    def test_batch_preprocessor(self, batch):
        for x_name in self.input_vars:
            batch[x_name] = labels_to_one_hot(batch[x_name], self.num_cats)
        batch['y'] = labels_to_one_hot(batch['y'], self.num_cats)
        return batch

    def pred_xs_preprocessor(self, xs):
        for x_name in self.input_vars:
            xs[x_name] = labels_to_one_hot(xs[x_name], self.num_cats)
        return xs
