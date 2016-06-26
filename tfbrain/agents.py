from collections import deque

import random

import numpy as np

import tensorflow as tf

# from memory_profiler import profile

from .helpers import get_all_params, \
    get_all_params_copies, set_all_params_ops, \
    set_all_net_params_ops
from .memory import ExperienceReplay


class DQNAgent(object):
    def __init__(self, hyperparams, q_model, q_trainer, param_fnm):
        self.hyperparams = hyperparams
        self.q_model = q_model
        self.q_trainer = q_trainer
        self.param_fnm = param_fnm

    def set_actions(self, actions):
        self.actions = actions
        self.q_model.set_num_actions(len(actions))

    def set_state_shape(self, state_shape):
        self.state_shape = state_shape
        self.q_model.set_state_shape(state_shape)

    def build(self):
        print('Building agent ...')
        self.q_trainer.build(
            train_mask_shape=(None, len(self.actions)))
        self.sess = self.q_trainer.sess
        self.build_vars()
        epsilon = self.hyperparams['epsilon']
        self.epsilon = epsilon[0]
        self.epsilon_step = (epsilon[1] - epsilon[0]) / epsilon[2]
        self.eval_epsilon = self.hyperparams['eval_epsilon']
        self.training = True
        self.recent_train_q = deque(
            maxlen=self.hyperparams['num_recent_frames'])
        self.recent_eval_q = deque(
            maxlen=self.hyperparams['num_recent_frames'])
        self.experience_replay = ExperienceReplay(self.hyperparams)
        self.experience_replay.build()

    def build_vars(self):
        self.model_params = get_all_params(self.q_model.net)
        self.target_model_params = get_all_params_copies(self.model_params)
        self.sess.run(tf.initialize_all_variables())
        self.update_target_weights_ops = set_all_params_ops(
            self.target_model_params, self.model_params,
            sess=self.sess)
        self.load_target_weights_ops = set_all_net_params_ops(
            self.q_model.net, self.target_model_params,
            sess=self.sess)
        self.load_model_weights_ops = set_all_net_params_ops(
            self.q_model.net, self.model_params,
            sess=self.sess)

    def save_params(self):
        self.q_model.save_params(self.param_fnm,
                                 sess=self.sess)

    def update_target_weights(self):
        for op in self.update_target_weights_ops:
            self.sess.run(op)

    def load_target_weights(self):
        for op in self.load_target_weights_ops:
            self.sess.run(op)

    def load_model_weights(self):
        for op in self.load_model_weights_ops:
            self.sess.run(op)

    def compute_single_state_preds(self, single_state):
        xs = {'state': np.reshape(single_state, (1,) + single_state.shape)}
        preds = self.q_model.compute_preds(xs)
        return preds

    def choose_greedy_action(self, state):
        preds = self.compute_single_state_preds(state)
        if self.training:
            self.recent_train_q.append(preds.max())
        else:
            self.recent_eval_q.append(preds.max())
        greedy_ind = np.argmax(preds)
        return self.actions[greedy_ind]

    def display_from_action(self, frame_num):
        if frame_num % self.hyperparams['display_freq'] == 0:
            if self.training:
                self.display_train_update()
            else:
                self.display_eval_update()

    def choose_random_action(self):
        return random.choice(self.actions)

    def choose_train_action(self, state):
        if self.epsilon > self.hyperparams['epsilon'][1]:
            self.epsilon += self.epsilon_step
        if random.random() < self.epsilon:
            return self.choose_random_action()
        else:
            return self.choose_greedy_action(state)

    def choose_eval_action(self, state):
        if random.random() < self.eval_epsilon:
            return self.choose_random_action()
        else:
            return self.choose_greedy_action(state)

    def choose_action(self, state, frame_num):
        self.display_from_action(frame_num)
        if frame_num < self.hyperparams['init_explore_len']:
            return self.choose_random_action()
        elif self.training:
            return self.choose_train_action(state)
        else:
            return self.choose_eval_action(state)

    def have_experience(self, experience):
        if self.training:
            # state, action, reward, next_state = experience
            self.experience_replay.add_experience(experience)

    def gen_train_data(self):
        batch_size = self.hyperparams['batch_size']
        num_experiences = batch_size
        experiences = self.experience_replay.sample(batch_size)

        self.load_target_weights()

        reward_discount = self.hyperparams['reward_discount']

        train_xs = {'state': np.zeros((num_experiences,) + self.state_shape)}
        train_y = np.zeros((num_experiences, len(self.actions)))
        train_mask = np.zeros((num_experiences, len(self.actions)))
        for experience_num, (state, action, reward, next_state) in \
                enumerate(experiences):
            train_xs['state'][experience_num] = state
            preds = self.compute_single_state_preds(next_state)
            exp_returns = preds.max()
            train_y[experience_num, action] = \
                reward + reward_discount * exp_returns
            train_mask[experience_num, action] = 1
        self.load_model_weights()
        return train_xs, train_y, train_mask

    def train_net(self, display_interval=None):
        if len(self.experience_replay) >= \
                self.hyperparams['batch_size']:
            train_xs, train_y, train_mask = self.gen_train_data()
            num_updates = self.hyperparams['updates_per_iter']
            self.q_trainer.train(
                train_xs, train_y, None, None, train_mask,
                build=False, num_updates=num_updates,
                display_interval=display_interval)

    def start_eval_mode(self):
        self.training = False

    def end_eval_mode(self):
        self.training = True

    def display_train_update(self):
        print('Epsilon: %f' % self.epsilon)
        if len(self.recent_train_q) > 0:
            print('Avg recent pred q: %f' %
                  (sum(self.recent_train_q) / len(self.recent_train_q)))

    def display_eval_update(self):
        print('Epsilon: %f' % self.eval_epsilon)
        if len(self.recent_eval_q) > 0:
            print('Avg recent pred q: %f' %
                  (sum(self.recent_eval_q) / len(self.recent_eval_q)))

    def learn(self, frame_num):
        if frame_num % self.hyperparams['target_update_freq'] == 0:
            self.update_target_weights()
        if frame_num % self.hyperparams['update_freq'] == 0:
            if frame_num % self.hyperparams['display_freq'] == 0:
                self.train_net(display_interval=100)
            else:
                self.train_net()
