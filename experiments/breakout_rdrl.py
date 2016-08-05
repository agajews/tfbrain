import tfbrain as tb
import numpy as np

from tasks.rl import AtariTask


class IdenticalCnnBuilder(object):

    def __init__(self):
        self.cnn = None

    def get_cnn(self, state_shape):
        if self.cnn is None:
            self.cnn = self.create_cnn(state_shape)
        return self.cnn

    def create_cnn(self, state_shape):
        state = tb.ly.InputLayer(shape=(None,) + state_shape,
                                 dtype=tb.float32,
                                 name='state')
        net = state
        # net = tb.ly.ReshapeLayer(i_state, (-1,) + screen_size + (state_len,))
        net = tb.ly.Conv2DLayer(net, (8, 8), 32, inner_strides=(4, 4),
                                pad='VALID',
                                W_b_init=tb.init.dqn(),
                                name='action_conv')
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (4, 4), 64, inner_strides=(2, 2),
                                pad='VALID',
                                W_b_init=tb.init.dqn())
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (3, 3), 64, inner_strides=(1, 1),
                                pad='VALID',
                                W_b_init=tb.init.dqn())
        net = tb.ly.FlattenLayer(net)

        return net


cnn = IdenticalCnnBuilder()


class ActionModel(tb.RLModel):

    def create_net(self, trainable=True):
        i_state = tb.ly.InputLayer(shape=(None, None) + self.state_shape,
                                   dtype=tb.float32,
                                   name='state')
        # net = tb.ly.ScaleLayer(i_state, 1/255.0)

        net = tb.ly.NetOnSeq(i_state, tb.clone_keep_params(
            cnn.get_cnn(self.state_shape), trainable=False))

        net = tb.ly.LSTMLayer(net, 512,
                              W_b_init=tb.init.dqn())
        net = tb.ly.SeqSliceLayer(net, col=-1)

        net = tb.ly.FullyConnectedLayer(net, 512,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)

        # net = tb.ly.DropoutLayer(net, 0.5)

        net = tb.ly.FullyConnectedLayer(net,
                                        self.num_actions,
                                        nonlin=tb.nonlin.softmax,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)
        return net


class StateModel(tb.RLModel):
    def create_net(self, trainable=True):
        rollout_len = self.hyperparams['value_rollout_length']
        i_state = tb.ly.InputLayer(shape=(None,) +
                                   self.state_shape[:-1] + (rollout_len,),
                                   dtype=tb.float32,
                                   name='state')
        # net = tb.ly.ScaleLayer(i_state, 1/255.0)
        # i_state = tb.ly.SqueezeLayer(i_state, 4)
        # i_state = tb.ly.TransposeLayer(i_state, [0, 2, 3, 1])
        i_state = tb.clone_keep_params(
            cnn.get_cnn(self.state_shape),
            passthrough_vars={'state': tb.get_output(i_state)})
        # net = tb.ly.NetOnSeq(i_state, tb.clone_keep_params(
        #     cnn.get_cnn(self.state_shape)))
        # net = tb.ly.LSTMLayer(net, 512,
        #                       W_b_init=tb.init.dqn())
        # net = tb.ly.SeqSliceLayer(net, col=-1)

        # i_next_state = tb.ly.ScaleLayer(i_next_state, 1/255.0)

        i_action = tb.ly.InputLayer(shape=(None, self.num_actions),
                                    dtype=tb.float32,
                                    name='action')
        i_action = tb.ly.FullyConnectedLayer(i_action, 256,
                                             W_b_init=tb.init.dqn(),
                                             trainable=trainable)

        net = tb.ly.MergeLayer([i_state, i_action])
        net = tb.ly.FullyConnectedLayer(net, 512,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)

        # net = tb.ly.DropoutLayer(net, 0.5)

        net = tb.ly.FullyConnectedLayer(net,
                                        np.prod(self.state_shape),
                                        nonlin=tb.nonlin.identity,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)

        # net = tb.ly.ReshapeLayer(net, (-1,) + self.state_shape)
        return net


class RewardModel(tb.RLModel):
    def create_net(self, trainable=True):
        print('State shape:')
        print(self.state_shape)
        rollout_len = self.hyperparams['value_rollout_length']
        i_state = tb.ly.InputLayer(shape=(None,) +
                                   self.state_shape[:-1] + (rollout_len,),
                                   dtype=tb.float32,
                                   name='state')
        i_state = tb.clone_keep_params(
            cnn.get_cnn(self.state_shape),
            passthrough_vars={'state': tb.get_output(i_state)})
        # net = tb.ly.ScaleLayer(i_state, 1/255.0)
        # i_state = tb.ly.SqueezeLayer(i_state, 4)
        # i_state = tb.ly.TransposeLayer(i_state, [0, 2, 3, 1])
        # i_state = tb.clone_keep_params(self.state_shape, net=i_state)
        # net = tb.ly.NetOnSeq(i_state, tb.clone_keep_params(
        #     cnn.get_cnn(self.state_shape)))
        # net = tb.ly.LSTMLayer(net, 512,
        #                       W_b_init=tb.init.dqn())
        # net = tb.ly.SeqSliceLayer(net, col=-1)

        # i_next_state = tb.ly.ScaleLayer(i_next_state, 1/255.0)

        # i_next_state = tb.ly.InputLayer(shape=(None,) + self.state_shape,
        #                                 dtype=tb.float32,
        #                                 name='next_state')
        # i_next_state = tb.clone_keep_params(
        #     cnn.get_cnn(self.state_shape),
        #     passthrough_vars={'state': tb.get_output(i_next_state)})

        i_action = tb.ly.InputLayer(shape=(None, self.num_actions),
                                    name='action')
        i_action = tb.ly.FullyConnectedLayer(i_action, 128,
                                             W_b_init=tb.init.dqn())

        net = tb.ly.MergeLayer([i_state,
                                i_action])

        net = tb.ly.FullyConnectedLayer(net, 512,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)

        # net = tb.ly.DropoutLayer(net, 0.5)

        net = tb.ly.FullyConnectedLayer(net, 1,
                                        nonlin=tb.nonlin.identity,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)
        return net


class ValueModel(tb.DQNModel):
    def create_net(self, trainable=True):
        rollout_len = self.hyperparams['value_rollout_length']
        print(self.state_shape)
        i_state = tb.ly.InputLayer(shape=(None,) +
                                   self.state_shape[:-1] + (rollout_len,),
                                   dtype=tb.float32,
                                   name='state')
        print('value input shape')
        print(i_state.get_output_shape())
        if trainable:
            i_state = tb.clone_keep_params(
                cnn.get_cnn(self.state_shape),
                passthrough_vars={'state': tb.get_output(i_state)})
        else:
            i_state = tb.clone(
                cnn.get_cnn(self.state_shape),
                passthrough_vars={'state': tb.get_output(i_state)})
        print('value input shape 2')
        print(tb.get_input_vars(i_state)['state'].get_shape())
        # net = tb.ly.ScaleLayer(i_state, 1/255.0)
        # i_state = tb.ly.SqueezeLayer(i_state, 4)
        # i_state = tb.ly.TransposeLayer(i_state, [0, 2, 3, 1])
        # net = tb.clone_keep_params(self.state_shape, net=i_state)
        # net = tb.ly.NetOnSeq(i_state, tb.clone_keep_params(
        #     cnn.get_cnn(self.state_shape)))
        # net = tb.ly.LSTMLayer(net, 512,
        #                       W_b_init=tb.init.dqn())
        # net = tb.ly.SeqSliceLayer(net, col=-1)

        net = tb.ly.FullyConnectedLayer(i_state, 512,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)

        # net = tb.ly.DropoutLayer(net, 0.5)

        net = tb.ly.FullyConnectedLayer(net, 1,
                                        nonlin=tb.nonlin.identity,
                                        trainable=trainable)

        return net


def train_rdrl():
    hyperparams = {'batch_size': 32,
                   'init_explore_len': 500000,
                   'num_mega_updates': 100000,
                   # 'init_model_train': 500000,
                   # 'init_explore_len': 50,
                   'learning_rate': 0.05,
                   # 'grad_momentum': 0.0,
                   'grad_decay': 0.95,
                   'grad_epsilon': 0.01,
                   # 'grad_norm_clip': 5,
                   'epsilon': (1.0, 0.1, 1000000),
                   'frame_skip': 4,
                   'reward_discount': 0.99,
                   'display_freq': 100,
                   'updates_per_model_iter': 1,
                   'updates_per_iter': 1,
                   # 'trains_per_action_train': 500,
                   'train_freq': 16,
                   'action_train_freq': 16,
                   # 'action_train_freq': 10000,
                   'frames_per_epoch': 100000,
                   # 'frames_per_epoch': 250,
                   'frames_per_eval': 50000,
                   # 'screen_resize': (110, 84),
                   'experience_replay_len': 4000000,
                   'update_target_freq': 20000,
                   # 'cache_size': int(2e4),
                   'state_len': 1,
                   # 'num_frames': 10000000,
                   # 'save_freq': 100000,
                   # 'eval_freq': 10,
                   'num_epochs': 200,  # 1e7 frames
                   'show_screen': False,
                   'rollout_length': 4,
                   'value_rollout_length': 4,
                   'eval_epsilon': 0.05,
                   'action_train_scale': 5,
                   'num_recent_episodes': 100,
                   'num_recent_steps': 10000}
    action_model = ActionModel(hyperparams)
    action_optim = tb.RMSPropOptim(hyperparams)

    state_model = StateModel(hyperparams)
    state_optim = tb.RMSPropOptim(hyperparams)
    state_loss = tb.MSE(hyperparams)

    reward_model = RewardModel(hyperparams)
    reward_optim = tb.RMSPropOptim(hyperparams)
    reward_loss = tb.MSE(hyperparams)

    value_model = ValueModel(hyperparams)
    value_optim = tb.RMSPropOptim(hyperparams)
    value_loss = tb.MSE(hyperparams)

    # q_trainer = tb.Trainer(q_model, hyperparams, loss, optim, evaluator)
    agent = tb.RDRLAgent(hyperparams,
                         action_model, action_optim,
                         state_model, state_loss, state_optim,
                         reward_model, reward_loss, reward_optim,
                         value_model, value_loss, value_optim,
                         'params/breakout_rdrl.json')
    task = AtariTask(hyperparams, 'data/roms/breakout.bin')
    trainer = tb.RLTrainer(hyperparams, agent, task)
    trainer.train_by_epoch()

if __name__ == '__main__':
    train_rdrl()
