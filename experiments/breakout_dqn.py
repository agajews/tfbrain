import tfbrain as tb

from tasks.rl import AtariTask


class DQNModel(tb.Model):

    def set_state_shape(self, state_shape):
        self.state_shape = state_shape

    def set_num_actions(self, num_actions):
        self.num_actions = num_actions

    def build_net(self):
        state_len = self.hyperparams['state_len']
        screen_size = self.state_shape[1:3]
        i_state = tb.ly.InputLayer(shape=(None,) + self.state_shape)
        net = tb.ly.ReshapeLayer(i_state, (-1,) + screen_size + (state_len,))
        net = tb.ly.Conv2DLayer(net, (4, 4), 16, inner_strides=(2, 2))
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (4, 4), 32, inner_strides=(2, 2))
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.FlattenLayer(net)
        # net = tb.ly.MergeLayer(net, axis=1)
        net = tb.ly.FullyConnectedLayer(net, 256)
        # net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net,
                                        self.num_actions,
                                        nonlin=tb.nonlin.identity)
        self.net = net
        self.input_vars = {'state': i_state.placeholder}


def train_dqn():
    hyperparams = {'batch_size': 32,
                   'init_explore_len': 50000,
                   'learning_rate': 0.00025,
                   'grad_norm_clip': 5,
                   'epsilon': (1.0, 0.1, int(5e5)),
                   'frame_skip': 4,
                   'reward_discount': 0.99,
                   'target_update_freq': 10000,
                   'display_freq': 100,
                   'updates_per_iter': 1,
                   'update_freq': 4,
                   'screen_resize': (110, 84),
                   'experience_replay_len': int(4e4),
                   'cache_size': int(2e4),
                   'state_len': 4,
                   'num_frames': int(1e7),
                   'save_freq': 100000,
                   'eval_freq': 10,
                   'eval_epsilon': 0.05,
                   'num_recent_rewards': 100,
                   'num_recent_frames': int(1e4)}
    q_model = DQNModel(hyperparams)
    loss = tb.MSE(hyperparams)
    acc = None
    evaluator = tb.Evaluator(hyperparams, loss, acc)
    optim = tb.RMSPropOptim(hyperparams)
    q_trainer = tb.Trainer(q_model, hyperparams, loss, optim, evaluator)
    agent = tb.DQNAgent(hyperparams, q_model, q_trainer,
                        'params/breakout_dqn.json')
    task = AtariTask(hyperparams, 'data/roms/breakout.bin')
    trainer = tb.RLTrainer(hyperparams, agent, task)
    trainer.train()

if __name__ == '__main__':
    train_dqn()
