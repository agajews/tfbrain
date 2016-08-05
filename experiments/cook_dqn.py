import tfbrain as tb

from tasks import CookingTask


class CookModel(tb.DQNModel):

    def create_net(self, trainable=True):
        # state_len = self.hyperparams['state_len']
        # screen_size = self.state_shape[1:3]
        # print(screen_size)
        i_state = tb.ly.InputLayer(shape=(None,) + self.state_shape,
                                   dtype=tb.float32,
                                   name='state')
        # net = i_state
        net = tb.ly.ScaleLayer(i_state, 1/255.0)
        # net = tb.ly.ReshapeLayer(i_state, (-1,) + screen_size + (state_len,))
        net = tb.ly.Conv2DLayer(net, (8, 8), 32, inner_strides=(4, 4),
                                W_b_init=tb.init.dqn(),
                                pad='VALID',
                                trainable=trainable)
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (4, 4), 64, inner_strides=(2, 2),
                                W_b_init=tb.init.dqn(),
                                pad='VALID',
                                trainable=trainable)
        # curr_net = tb.ly.MaxPool2DLayer(curr_net, (2, 2),
        #                                 inner_strides=(2, 2))
        net = tb.ly.Conv2DLayer(net, (3, 3), 64, inner_strides=(1, 1),
                                W_b_init=tb.init.dqn(),
                                pad='VALID',
                                trainable=trainable)
        net = tb.ly.FlattenLayer(net)
        # net = tb.ly.MergeLayer(net, axis=1)
        net = tb.ly.FullyConnectedLayer(net, 512,
                                        W_b_init=tb.init.dqn(),
                                        trainable=trainable)
        # net = tb.ly.DropoutLayer(net, 0.5)
        net = tb.ly.FullyConnectedLayer(net,
                                        self.num_actions,
                                        W_b_init=tb.init.dqn(),
                                        nonlin=tb.nonlin.softmax,
                                        trainable=trainable)
        return net


def train_dqn():
    hyperparams = {'batch_size': 32,
                   'init_explore_len': 50000,
                   # 'init_explore_len': 50,
                   'learning_rate': 0.00025,
                   # 'grad_momentum': 0.0,
                   'grad_decay': 0.95,
                   'grad_epsilon': 0.01,
                   # 'grad_norm_clip': 5,
                   'epsilon': (1.0, 0.1, 1000000),
                   'frame_skip': 10,
                   'num_recent_feats': 25,
                   'steps_per_episode': 150,
                   'reward_discount': 0.99,
                   'show_screen': True,
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
                   'joint_vel': 0.5,
                   # 'num_frames': 10000000,
                   # 'save_freq': 100000,
                   # 'eval_freq': 10,
                   'num_epochs': 200,  # 1e7 frames
                   'eval_epsilon': 0.05,
                   'num_recent_episodes': 100,
                   'num_recent_steps': 10000}
    q_model = CookModel(hyperparams)
    loss = tb.MSE(hyperparams)
    optim = tb.RMSPropOptim(hyperparams)
    # q_trainer = tb.Trainer(q_model, hyperparams, loss, optim, evaluator)
    agent = tb.DQNAgent(hyperparams, q_model, optim, loss,
                        'params/cook_dqn.json')
    task = CookingTask(hyperparams)
    trainer = tb.RLTrainer(hyperparams, agent, task,
                           load_first=True)
    trainer.train_by_epoch()

if __name__ == '__main__':
    train_dqn()
