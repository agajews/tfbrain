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
        net = tb.ly.Conv2DLayer(net, (8, 8), 16, inner_strides=(4, 4),
                                W_b_init=tb.init.dqn(),
                                pad='VALID',
                                trainable=trainable)
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (4, 4), 32, inner_strides=(2, 2),
                                W_b_init=tb.init.dqn(),
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
        net = tb.ly.FullyConnectedLayer(net, 256,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)
        # net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net,
                                        self.num_actions,
                                        W_b_init=tb.init.dqn(),
                                        nonlin=tb.nonlin.identity,
                                        trainable=trainable)
        return net


def train_dqn():
    hyperparams = {'batch_size': 32,
                   'learning_rate': (0.002, 0.00025, 16000000),
                   'grad_decay': 0.99,
                   'grad_epsilon': 0.01,
                   'epsilon': [(1.0, 0.1, 4000000, 0.4),
                               (1.0, 0.01, 4000000, 0.3),
                               (1.0, 0.5, 4000000, 0.3)],
                   # 'epsilon': (1, 0
                   # .1, 4000000),
                   'frame_skip': 4,
                   'reward_discount': 0.99,
                   'show_screen': False,
                   'display_freq': 100,
                   # 'updates_per_iter': 1000,
                   'updates_per_iter': 2500,
                   'num_threads': 16,
                   # 'update_freq': 4,
                   # 'frames_per_epoch': 5000,
                   'frames_per_epoch': 40000,
                   'episodes_per_eval': 32,
                   # 'frames_per_eval': 25000,
                   # 'frames_per_eval': 5000,
                   'state_len': 4,
                   'num_epochs': 10000,
                   'eval_epsilon': 0.05,
                   'num_recent_episodes': 100,
                   'tmax': 5,
                   'eval_freq': 3,
                   'grad_norm_clip': 40.0,
                   'num_recent_steps': 10000}
    q_model = BreakoutModel(hyperparams)
    loss = tb.MSE(hyperparams)
    optim = tb.RMSPropVarOptim(hyperparams)
    # q_trainer = tb.Trainer(q_model, hyperparams, loss, optim, evaluator)
    agent = tb.AsyncEdistSNDQNV2Agent(
        hyperparams, q_model, optim, loss,
        'params/breakout_async_sndqn_nfix7.json')
    task = AsyncAtariTask(hyperparams, 'data/roms/breakout.bin')
    trainer = tb.AsyncSleepSaveTrainer(hyperparams, agent, task,
                                       load_first=False)
    trainer.train_by_epoch()

if __name__ == '__main__':
    train_dqn()
