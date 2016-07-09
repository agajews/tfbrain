import json

from tfbrain.helpers import get_output, \
    create_x_feed_dict, create_supp_test_feed_dict, \
    get_all_net_params_values, get_all_params

from tasks import labels_to_one_hot


class Model(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def build_net(self):
        raise NotImplementedError()

    def get_net(self):
        return self.net

    def get_input_vars(self):
        return self.input_vars

    def update_hyperparams(self, update_dict):
        self.hyperparams.update(update_dict)

    def train_batch_preprocessor(self, batch):
        return batch

    def test_batch_preprocessor(self, batch):
        return batch

    def pred_xs_preprocessor(self, xs):
        return xs

    def setup_net(self):
        self.build_net()
        self.y_hat = get_output(self.get_net())

    def compute_preds(self, xs, sess):
        xs = self.pred_xs_preprocessor(xs)
        feed_dict = create_x_feed_dict(self.input_vars, xs)
        feed_dict.update(create_supp_test_feed_dict(self))
        preds = self.y_hat.eval(feed_dict=feed_dict,
                                session=sess)
        return preds

    def save_params(self, fnm, sess):
        self._save_params(self.get_net(), fnm, sess)

    def _save_params(self, net, fnm, sess):
        params_values = get_all_net_params_values(net, sess=sess)
        for layer_name in params_values.keys():
            for param_name in params_values[layer_name].keys():
                param = params_values[layer_name][param_name]
                params_values[layer_name][param_name] = param.tolist()
        with open(fnm, 'w') as f:
            json.dump(params_values, f)

    def load_params(self, fnm, sess):
        self._load_params(self.get_net(), fnm, sess)

    def _load_params(self, net, fnm, sess):
        dest_params = get_all_params(net)
        with open(fnm, 'r') as f:
            src_params = json.loads(f.read())
        for layer_name in dest_params.keys():
            for param_name in dest_params[layer_name].keys():
                dest_param = dest_params[layer_name][param_name]
                src_param = src_params[layer_name][param_name]
                sess.run(dest_param.assign(src_param))


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
