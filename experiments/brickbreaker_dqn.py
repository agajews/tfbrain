import tfbrain as tb

from tasks.rl import AtariTask


class DQNModel(tb.Model):

    def build_net(self):
        state_shape = self.hyperparams['state_shape']
        num_actions = self.hyperparams['num_actions']
        state_len = self.hyperparams['state_len']
        screen_size = state_shape[1:3]
        i_state = tb.ly.InputLayer(shape=(None,) + state_shape)
        net = tb.ly.ReshapeLayer(i_state, (-1,) + screen_size + (state_len,))
        net = tb.ly.Conv2DLayer(net, (4, 4), 16, inner_strides=(2, 2))
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2), inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (4, 4), 32, inner_strides=(2, 2))
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2), inner_strides=(2, 2))
        net = tb.ly.FlattenLayer(net)
        # net = tb.ly.MergeLayer(net, axis=1)
        net = tb.ly.FullyConnectedLayer(net, 256)
        # net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net,
                                        num_actions,
                                        nonlin=tb.nonlin.identity)
        self.net = net
        self.input_vars = {'state': i_state.placeholder}


def train_dqn():
    hyperparams = {'batch_size': 32,
                   'learning_rate': 1e-5,
                   'grad_norm_clip': 5,
                   'epsilon': 0.1,
                   'frame_skip': 4,
                   'reward_discount': 0.99,
                   'target_update_freq': 100,
                   'updates_per_iter': 1,
                   'screen_resize': (110, 84),
                   'experience_replay_len': int(1e6),
                   'state_len': 4,
                   'num_frames': int(1e7),
                   'save_freq': 100000,
                   'num_recent_rewards': 100}
    q_model = DQNModel(hyperparams)
    loss = tb.MSE(hyperparams)
    acc = None
    evaluator = tb.Evaluator(hyperparams, loss, acc)
    optim = tb.AdamOptim(hyperparams)
    q_trainer = tb.Trainer(q_model, hyperparams, loss, optim, evaluator)
    agent = tb.DQNAgent(hyperparams, q_model, q_trainer)
    task = AtariTask(hyperparams, 'data/roms/breakout.bin')
    trainer = tb.RLTrainer(hyperparams, agent, task)
    trainer.train()

if __name__ == '__main__':
    train_dqn()
