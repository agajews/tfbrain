import json

from tfbrain.helpers import get_output, \
    create_x_feed_dict, create_supp_test_feed_dict, \
    get_all_net_params_values, get_all_params, \
    get_input_vars

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
        self.input_vars = get_input_vars(self.get_net())
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


class RLModel(Model):

    def set_state_shape(self, state_shape):
        self.state_shape = state_shape

    def set_num_actions(self, num_actions):
        self.num_actions = num_actions

    def build_net(self):
        self.net = self.create_net()


class ACModel(RLModel):

    def build_net(self):
        self.state_processor = self.create_state_processor()
        self.policy = self.create_policy()
        self.value = self.create_value()

    def setup_net(self):
        self.build_net()
        self.policy_y_hat = get_output(self.policy)
        self.value_y_hat = get_output(self.value)
        self.policy_input_vars = get_input_vars(self.policy)
        self.value_input_vars = get_input_vars(self.value)

    def compute_policy_preds(self, xs, sess):
        return self._compute_preds(self.policy_y_hat, self.policy_input_vars,
                                   xs, sess)

    def compute_value_preds(self, xs, sess):
        return self._compute_preds(self.value_y_hat, self.value_input_vars,
                                   xs, sess)

    def _compute_preds(self, y_hat, input_vars, xs, sess):
        xs = self.pred_xs_preprocessor(xs)
        feed_dict = create_x_feed_dict(input_vars, xs)
        # feed_dict.update(create_supp_test_feed_dict(self))
        preds = y_hat.eval(feed_dict=feed_dict,
                           session=sess)
        return preds

    def load_params(self, fnm, sess):
        self.load_policy_params(fnm, sess)
        self.load_value_params(fnm, sess)

    def save_params(self, fnm, sess):
        self.save_policy_params(fnm, sess)
        self.save_value_params(fnm, sess)

    def load_policy_params(self, fnm, sess):
        self._load_params(self.policy, fnm, sess)

    def save_policy_params(self, fnm, sess):
        self._save_params(self.policy, fnm, sess)

    def load_value_params(self, fnm, sess):
        self._load_params(self.value, fnm, sess)

    def save_value_params(self, fnm, sess):
        self._save_params(self.value, fnm, sess)


class DQNModel(RLModel):

    def get_target_net(self):
        return self.target_net

    def get_target_input_vars(self):
        return self.target_input_vars

    def setup_net(self):
        self.build_net()
        self.y_hat = get_output(self.get_net())
        self.target_y_hat = get_output(self.get_target_net())

    def compute_target_preds(self, xs, sess):
        xs = self.pred_xs_preprocessor(xs)
        feed_dict = create_x_feed_dict(self.target_input_vars, xs)
        feed_dict.update(create_supp_test_feed_dict(self))
        preds = self.target_y_hat.eval(feed_dict=feed_dict,
                                       session=sess)
        return preds

    def build_net(self):
        self.net = self.create_net()
        self.input_vars = get_input_vars(self.net)
        self.target_net = self.create_net(trainable=False)
        self.target_input_vars = get_input_vars(self.target_net)

    def load_target_params(self, fnm, sess):
        self._load_params(self.get_target_net(), fnm, sess)

    def save_target_params(self, fnm, sess):
        self._save_params(self.get_target_net(), fnm, sess)
