from collections import deque

import copy

import random

import numpy as np

import json

import tensorflow as tf

from .helpers import get_all_params, \
    get_all_params_copies, set_all_params_ops, \
    set_all_net_params_ops, get_all_params_values, \
    ShelfDeque


class DQNAgent(object):
    def __init__(self, hyperparams, q_model, q_trainer):
        self.hyperparams = hyperparams
        self.q_model = q_model
        self.q_trainer = q_trainer
        # self.experiences = deque(maxlen=hyperparams['experience_replay_len'])
        self.experiences = ShelfDeque(
            'data/experiences.db',
            maxlen=hyperparams['experience_replay_len'])

    def set_actions(self, actions):
        self.actions = actions
        self.q_model.hyperparams['num_actions'] = len(actions)

    def set_state_shape(self, state_shape):
        self.state_shape = state_shape
        self.q_model.hyperparams['state_shape'] = state_shape

    def build(self):
        print('Building agent ...')
        self.q_trainer.build(
            train_mask_shape=(None, len(self.actions)))
        self.sess = self.q_trainer.sess
        self.build_vars()
        epsilon = self.hyperparams['epsilon']
        self.epsilon = epsilon[0]
        self.epsilon_step = (epsilon[1] - epsilon[0]) / epsilon[2]
        self.training = True
        self.recent_pred_q = deque(
            maxlen=self.hyperparams['num_recent_rewards'])

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
        params_values = get_all_params_values(self.model_params,
                                              sess=self.sess)
        for layer_name in params_values.keys():
            for param_name in params_values[layer_name].keys():
                param = params_values[layer_name][param_name]
                params_values[layer_name][param_name] = param.tolist()
        with open('params/brickbreaker_dqn.json', 'w') as f:
            json.dump(params_values, f)

    def update_target_weights(self):
        # print('Target weights:')
        # print(get_all_params_values(self.target_model_params,
        #                             sess=self.sess)['fc_l7'])
        # print('Model weights:')
        # print(get_all_params_values(self.model_params,
        #                             sess=self.sess)['fc_l7'])
        for op in self.update_target_weights_ops:
            self.sess.run(op)
        # print('Updated target weights:')
        # print(get_all_params_values(self.target_model_params,
        #                             sess=self.sess)['fc_l7'])

    def load_target_weights(self):
        # print('Target weights:')
        # print(get_all_params_values(self.target_model_params,
        #                             sess=self.sess)['fc_l7'])
        # print('Model weights:')
        # print(get_all_params_values(self.model_params,
        #                             sess=self.sess)['fc_l7'])
        for op in self.load_target_weights_ops:
            self.sess.run(op)

    def load_model_weights(self):
        # print('Target weights:')
        # print(get_all_params_values(self.target_model_params,
        #                             sess=self.sess)['fc_l7'])
        # print('Model weights:')
        # print(get_all_params_values(self.model_params,
        #                             sess=self.sess)['fc_l7'])
        for op in self.load_model_weights_ops:
            self.sess.run(op)

    # def update_pred_weights(self):
    #     self.pred_weights = get_all_params_values(self.q_model.net,
    #                                               sess=self.sess)

    # def load_pred_weights(self):
    #     self.target_weights = get_all_params_values(self.q_model.net,
    #                                                 sess=self.sess)
    #     set_all_params_values(self.q_model.net, self.pred_weights,
    #                           sess=self.sess)

    # def load_target_weights(self):
    #     set_all_params_values(self.q_model.net, self.target_weights,
    #                           sess=self.sess)

    def compute_single_state_preds(self, single_state):
        xs = {'state': np.reshape(single_state, (1,) + single_state.shape)}
        preds = self.q_model.compute_preds(xs)
        return preds

    def choose_action(self, state):
        if random.random() < self.epsilon:
            if self.epsilon > self.hyperparams['epsilon'][1]:
                self.epsilon += self.epsilon_step
            return random.choice(self.actions)
        else:
            preds = self.compute_single_state_preds(state)
            self.recent_pred_q.append(preds.max())
            greedy_ind = np.argmax(preds)
            return self.actions[greedy_ind]

    def have_experience(self, experience):
        self.experiences.append(experience)
        # print('Action: %s' % str(experience[1]))
        # print('Reward: %f' % experience[2])

    def gen_train_data(self):
        batch_size = self.hyperparams['batch_size']
        # experiences = random.sample(self.experiences, batch_size)
        experiences = self.experiences.sample(batch_size)

        # preds = self.compute_single_state_preds(experiences[0][3])
        # print('Model preds:')
        # print(preds)

        self.load_target_weights()

        # preds = self.compute_single_state_preds(experiences[0][3])
        # print('Target preds:')
        # print(preds)

        # self.update_target_weights()
        # preds = self.compute_single_state_preds(experiences[0][3])
        # print('Updated target preds:')
        # print(preds)

        reward_discount = self.hyperparams['reward_discount']
        num_experiences = len(experiences)
        train_xs = {'state': np.zeros((num_experiences,) + self.state_shape)}
        train_y = np.zeros((num_experiences, len(self.actions)))
        train_mask = np.zeros((num_experiences, len(self.actions)))
        for experience_num, experience in enumerate(experiences):
            state, action, reward, next_state = experience
            train_xs['state'][experience_num] = state
            preds = self.compute_single_state_preds(next_state)
            exp_returns = preds.max()
            train_y[experience_num, action] = \
                reward + reward_discount * exp_returns
            train_mask[experience_num, action] = 1
        self.load_model_weights()
        return train_xs, train_y, train_mask

    def train_net(self, display_interval=None):
        if len(self.experiences) >= self.hyperparams['batch_size']:
            train_xs, train_y, train_mask = self.gen_train_data()
            num_updates = self.hyperparams['updates_per_iter']
            self.q_trainer.train(
                train_xs, train_y, None, None, train_mask,
                build=False, num_updates=num_updates,
                display_interval=display_interval)

    def start_eval_mode(self):
        self.saved_epsilon_step = self.epsilon_step
        self.saved_epsilon = self.epsilon
        self.saved_recent_pred_q = copy.copy(self.recent_pred_q)
        self.epsilon_step = 0
        self.epsilon = self.hyperparams['eval_epsilon']
        self.training = False

    def end_eval_mode(self):
        self.epsilon_step = self.saved_epsilon_step
        self.epsilon = self.saved_epsilon
        self.recent_pred_q = self.saved_recent_pred_q
        self.training = True

    def learn(self, frame_num):
        if self.training and \
                frame_num % self.hyperparams['target_update_freq'] == 0:
            self.update_target_weights()
        if frame_num % self.hyperparams['display_freq'] == 0:
            if self.training:
                self.train_net(display_interval=100)
            print('Epsilon: %f' % self.epsilon)
            if len(self.recent_pred_q) > 0:
                print('Avg recent pred q: %f' %
                      (sum(self.recent_pred_q) / len(self.recent_pred_q)))
        else:
            if self.training:
                self.train_net()
