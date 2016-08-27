import tfbrain as tb

from tasks import AsyncAtariTask


class BreakoutModel(tb.ACModel):

    def create_state_processor(self):
        i_state = tb.ly.InputLayer(shape=(None,) + self.state_shape,
                                   dtype=tb.uint8,
                                   name='state')
        net = i_state
        net = tb.ly.ScaleLayer(i_state, 1/255.0)
        net = tb.ly.Conv2DLayer(net, (8, 8), 16, inner_strides=(4, 4),
                                W_b_init=tb.init.dqn(),
                                pad='VALID',
                                trainable=True)
        net = tb.ly.Conv2DLayer(net, (4, 4), 32, inner_strides=(2, 2),
                                W_b_init=tb.init.dqn(),
                                pad='VALID',
                                trainable=True)
        net = tb.ly.FlattenLayer(net)
        net = tb.ly.FullyConnectedLayer(net, 256,
                                        W_b_init=tb.init.dqn(),
                                        trainable=True)
        return net

    def create_policy(self, trainable=True):
        net = tb.ly.FullyConnectedLayer(self.state_processor,
                                        self.num_actions,
                                        W_b_init=tb.init.dqn(),
                                        nonlin=tb.nonlin.softmax,
                                        trainable=trainable)
        return net

    def create_value(self, trainable=True):
        net = tb.ly.FullyConnectedLayer(self.state_processor,
                                        1,
                                        W_b_init=tb.init.dqn(),
                                        nonlin=tb.nonlin.identity,
                                        trainable=trainable)
        return net


def train_snac():
    hyperparams = {'batch_size': 32,
                   'learning_rate': (0.002, 0.00025, 16000000),
                   'grad_decay': 0.99,
                   'grad_epsilon': 0.01,
                   # 'epsilon': [(1, 0.1, 4000000, 0.4),
                   #             (1, 0.01, 4000000, 0.3),
                   #             (1, 0.5, 4000000, 0.3)],
                   'epsilon': [(0.00, 0.00, 100, 1.0)],
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
                   'num_epochs': 4000,
                   'eval_epsilon': 0.05,
                   'num_recent_episodes': 100,
                   'tmax': 5,
                   'policy_clip': 1e-20,
                   'entropy_scale': 0.01,
                   'value_loss_scale': 0.5,
                   'eval_freq': 10,
                   'eval': False,
                   'prob_clip': 1e-6,
                   'grad_norm_clip': 40.0,
                   'num_recent_steps': 10000}
    ac_model = BreakoutModel(hyperparams)
    optim = tb.RMSPropVarOptim(hyperparams)
    # q_trainer = tb.Trainer(q_model, hyperparams, loss, optim, evaluator)
    agent = tb.AsyncSNACV2Agent(
        hyperparams, ac_model, optim,
        'params/breakout_async_snac_v2.json')
    task = AsyncAtariTask(hyperparams, 'data/roms/breakout.bin')
    trainer = tb.AsyncSleepSaveTrainer(hyperparams, agent, task,
                                       load_first=False)
    trainer.train_by_epoch()

if __name__ == '__main__':
    train_snac()
