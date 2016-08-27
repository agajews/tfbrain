import tfbrain as tb

from tasks import AsyncAtariTask


class BreakoutModel(tb.RLModel):

    def create_net(self, trainable=True):
        # state_len = self.hyperparams['state_len']
        # screen_size = self.state_shape[1:3]
        # print(screen_size)
        i_state = tb.ly.InputLayer(shape=(None,) + self.state_shape,
                                   dtype=tb.uint8,
                                   name='state')
        net = i_state
        net = tb.ly.ScaleLayer(i_state, 1/255.0)
        # net = tb.ly.ReshapeLayer(i_state, (-1,) + screen_size + (state_len,))
        net = tb.ly.Conv2DLayer(net, (8, 8), 32, inner_strides=(4, 4),
                                W_b_init=tb.init.dqn(),
                                # W_init=tb.init.truncated_normal(0.001),
                                # b_init=tb.init.constant(0.001),
                                pad='VALID',
                                trainable=trainable)
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (4, 4), 64, inner_strides=(2, 2),
                                W_b_init=tb.init.dqn(),
                                # W_init=tb.init.truncated_normal(0.001),
                                # b_init=tb.init.constant(0.001),
                                pad='VALID',
                                trainable=trainable)
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        # net = tb.ly.Conv2DLayer(net, (3, 3), 64, inner_strides=(1, 1),
        #                         W_b_init=tb.init.dqn(),
        #                         pad='VALID',
        #                         trainable=trainable)
        net = tb.ly.FlattenLayer(net)
        # net = tb.ly.MergeLayer(net, axis=1)
        net = tb.ly.FullyConnectedLayer(net, 1024,
                                        W_b_init=tb.init.dqn(),
                                        # W_init=tb.init.truncated_normal(0.001),
                                        # b_init=tb.init.constant(0.001),
                                        trainable=trainable)
        # net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net, self.num_actions,
                                        W_b_init=tb.init.dqn(),
                                        # W_init=tb.init.truncated_normal(0.001),
                                        # b_init=tb.init.constant(0.001),
                                        nonlin=tb.nonlin.identity,
                                        trainable=trainable)
        return net


def train_dqn():
    hyperparams = {'batch_size': 32,
                   'learning_rate': (0.5, 0.001, 4000000),
                   'grad_decay': 0.99,
                   'grad_epsilon': 0.01,
                   'epsilon': [(1, 0.1, 4000000, 0.4),
                               (1, 0.01, 4000000, 0.3),
                               (1, 0.5, 4000000, 0.3)],
                   # 'epsilon': (1, 0.1, 4000000),
                   'frame_skip': 4,
                   'reward_discount': 0.99,
                   'show_screen': False,
                   'display_freq': 100,
                   # 'updates_per_iter': 1000,
                   'updates_per_iter': 40000,
                   # 'init_frames': 20000,
                   # 'init_frames': 200000,
                   # 'init_updates': 20000,
                   # 'init_updates': 100000,
                   'num_threads': 16,
                   # 'update_freq': 4,
                   # 'frames_per_epoch': 5000,
                   'frames_per_epoch': 100000,
                   'episodes_per_eval': 32,
                   # 'episodes_per_eval': 16,
                   'state_len': 4,
                   'num_epochs': 400,
                   'eval_freq': 1,
                   'eval_epsilon': 0.05,
                   'num_recent_episodes': 100,
                   # 'tmax': 5,
                   'num_recent_steps': 10000}
    q_model = BreakoutModel(hyperparams)
    loss = tb.MSE(hyperparams)
    optim = tb.RMSPropVarOptim(hyperparams)
    # q_trainer = tb.Trainer(q_model, hyperparams, loss, optim, evaluator)
    agent = tb.AsyncSIDQNAgent(
        hyperparams, q_model, optim, loss,
        'params/breakout_async_sidqn_fix.json')
    task = AsyncAtariTask(hyperparams, 'data/roms/breakout.bin')
    trainer = tb.AsyncSleepTrainer(hyperparams, agent, task,
                                   load_first=False)
    trainer.train_by_epoch()

if __name__ == '__main__':
    train_dqn()
