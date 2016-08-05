import tfbrain as tb

from tasks.rl import AtariTask
from tfbrain.helpers import get_output, \
    create_x_feed_dict, create_supp_test_feed_dict


class DQNModel(tb.Model):

    def set_state_shape(self, state_shape):
        self.state_shape = state_shape

    def set_num_actions(self, num_actions):
        self.num_actions = num_actions

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

    def create_net(self, trainable=True):
        # state_len = self.hyperparams['state_len']
        # screen_size = self.state_shape[1:3]
        # print(screen_size)
        i_state = tb.ly.InputLayer(shape=(None,) + self.state_shape,
                                   dtype=tb.uint8)
        net = i_state
        net = tb.ly.ScaleLayer(i_state, 1/255.0)
        # net = tb.ly.ReshapeLayer(i_state, (-1,) + screen_size + (state_len,))
        net = tb.ly.Conv2DLayer(net, (8, 8), 32, inner_strides=(4, 4),
                                W_init=tb.init.dqn_weight(),
                                b_init=tb.init.dqn_bias(),
                                pad='VALID',
                                trainable=trainable)
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (4, 4), 64, inner_strides=(2, 2),
                                W_init=tb.init.dqn_weight(),
                                b_init=tb.init.dqn_bias(),
                                pad='VALID',
                                trainable=trainable)
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (3, 3), 64, inner_strides=(1, 1),
                                W_init=tb.init.dqn_weight(),
                                b_init=tb.init.dqn_bias(),
                                pad='VALID',
                                trainable=trainable)
        net = tb.ly.FlattenLayer(net)
        # net = tb.ly.MergeLayer(net, axis=1)
        net = tb.ly.FullyConnectedLayer(net, 512,
                                        W_init=tb.init.dqn_weight(),
                                        b_init=tb.init.dqn_bias(),
                                        trainable=trainable)
        # net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net,
                                        self.num_actions,
                                        W_init=tb.init.dqn_weight(),
                                        b_init=tb.init.dqn_bias(),
                                        nonlin=tb.nonlin.identity,
                                        trainable=trainable)
        return net, {'state': i_state.placeholder}

    def build_net(self):
        self.net, self.input_vars = self.create_net()
        self.target_net, self.target_input_vars = \
            self.create_net(trainable=False)

    def load_target_params(self, fnm, sess):
        self._load_params(self.get_target_net(), fnm, sess)

    def save_target_params(self, fnm, sess):
        self._save_params(self.get_target_net(), fnm, sess)


def train_dqn():
    hyperparams = {'batch_size': 32,
                   'init_explore_len': 30000,
                   # 'init_explore_len': 50,
                   'learning_rate': 0.00025,
                   # 'grad_momentum': 0.0,
                   'grad_decay': 0.95,
                   'grad_epsilon': 0.01,
                   # 'grad_norm_clip': 5,
                   'epsilon': (1.0, 0.1, 1000000),
                   'frame_skip': 4,
                   'reward_discount': 0.99,
                   'target_update_freq': 10000,
                   'display_freq': 25,
                   'updates_per_iter': 1,
                   'update_freq': 4,
                   'frames_per_epoch': 100000,
                   # 'frames_per_epoch': 250,
                   'frames_per_eval': 50000,
                   # 'screen_resize': (110, 84),
                   'experience_replay_len': 1000000,
                   # 'cache_size': int(2e4),
                   'state_len': 4,
                   # 'num_frames': 10000000,
                   # 'save_freq': 100000,
                   # 'eval_freq': 10,
                   'num_epochs': 200,  # 1e7 frames
                   'eval_epsilon': 0.05,
                   'num_recent_episodes': 100,
                   'num_recent_steps': 10000}
    q_model = DQNModel(hyperparams)
    loss = tb.MSE(hyperparams)
    optim = tb.RMSPropOptim(hyperparams)
    # q_trainer = tb.Trainer(q_model, hyperparams, loss, optim, evaluator)
    agent = tb.DQNAgent(hyperparams, q_model, optim, loss,
                        'params/montezuma_dqn.json')
    task = AtariTask(hyperparams, 'data/roms/montezuma_revenge.bin')
    trainer = tb.RLTrainer(hyperparams, agent, task, load_first=True)
    trainer.train_by_epoch()

if __name__ == '__main__':
    train_dqn()
