from collections import deque

from tfbrain.helpers import create_x_feed_dict, \
    create_y_feed_dict, create_supp_train_feed_dict, \
    create_supp_test_feed_dict, softmax, create_minibatch_iterator

import tensorflow as tf

import random

import numpy as np

# from memory_profiler import profile

from tfbrain.helpers import get_all_params, \
    set_all_params_ops
from tfbrain.memory import ExperienceReplay


class SDQNAgent(object):
    def __init__(self, hyperparams, q_model, optim, loss, param_fnm,
                 dtype=np.uint8):
        self.hyperparams = hyperparams
        self.q_model = q_model
        self.optim = optim
        self.loss = loss
        self.param_fnm = param_fnm
        self.dtype = dtype

    def start_episode(self):
        pass

    def set_actions(self, actions):
        self.actions = actions
        self.q_model.set_num_actions(len(actions))

    def set_state_shape(self, state_shape):
        self.state_shape = state_shape
        self.q_model.set_state_shape(state_shape)

    def prepare_epsilon(self):
        epsilon = self.hyperparams['epsilon']
        self.epsilon = epsilon[0]
        self.epsilon_step = (epsilon[1] - epsilon[0]) / epsilon[2]
        self.eval_epsilon = self.hyperparams['eval_epsilon']

    def prepare_debug_vars(self):
        self.target = None
        self.exp_returns = None
        self.greedy_ind = None
        self.exp_pred_returns = None
        self.zero_pred_returns = None

    def build(self):
        print('Building agent ...')
        self.q_model.setup_net()
        self.mask = tf.placeholder(shape=(None, len(self.actions)),
                                   dtype=tf.float32,
                                   name='mask')
        self.y = tf.placeholder(
            dtype=tf.float32,
            shape=(None,),
            name='expected_y')
        self.loss.build(self.q_model.y_hat, self.y, self.mask)
        self.train_step = self.optim.get_train_step(self.loss.loss)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        # self.sess.run(tf.initialize_variables(flatten_params(
        #     get_all_params(self.q_model.net))))

        self.prepare_epsilon()
        self.prepare_debug_vars()
        self.training = True
        self.recent_train_q = deque(
            maxlen=self.hyperparams['num_recent_steps'])
        self.recent_eval_q = deque(
            maxlen=self.hyperparams['num_recent_steps'])
        self.experience_replay = ExperienceReplay(self.hyperparams,
                                                  self.dtype)
        self.experience_replay.build()

        self.update_target_weights_ops = set_all_params_ops(
            get_all_params(self.q_model.get_target_net()),
            get_all_params(self.q_model.get_net()))
        self.update_target_weights()

    def save_params(self):
        self.q_model.save_params(self.param_fnm,
                                 sess=self.sess)

    def load_params(self):
        self.q_model.load_params(self.param_fnm,
                                 sess=self.sess)

    def save_target_params(self):
        self.q_model.save_target_params('params/breakout_dqn_target.json',
                                        sess=self.sess)

    def load_target_params(self):
        self.q_model.load_target_params('params/breakout_dqn_target.json',
                                        sess=self.sess)

    def update_target_weights(self):
        print('Updating target weights ...')
        self.sess.run(self.update_target_weights_ops[0:])

    def get_single_state_dict(self, single_state):
        xs = {'state': np.reshape(single_state, (1,) + single_state.shape)}
        return xs

    def compute_single_state_preds(self, single_state):
        xs = self.get_single_state_dict(single_state)
        preds = self.q_model.compute_preds(xs, sess=self.sess)
        return preds

    def compute_single_state_target_preds(self, single_state):
        xs = self.get_single_state_dict(single_state)
        preds = self.q_model.compute_target_preds(xs, sess=self.sess)
        return preds

    def choose_greedy_action(self, state):
        preds = self.compute_single_state_preds(state)
        self.preds = preds
        if self.training:
            self.recent_train_q.append(preds.max())
        else:
            self.recent_eval_q.append(preds.max())
        greedy_ind = np.argmax(np.squeeze(preds))
        self.greedy_ind = greedy_ind
        return preds

    def display_from_action(self, step_num):
        if step_num % self.hyperparams['display_freq'] == 0:
            if self.training:
                self.display_train_update(step_num)
            else:
                self.display_eval_update(step_num)

    def choose_random_action(self):
        return softmax(np.expand_dims(np.random.randn(len(self.actions)), 0))

    def update_epsilon(self):
        if self.epsilon > self.hyperparams['epsilon'][1]:
            self.epsilon += self.epsilon_step

    def choose_train_action(self, state):
        self.update_epsilon()
        if random.random() < self.epsilon:
            return self.choose_random_action()
        else:
            return self.choose_greedy_action(state)

    def choose_eval_action(self, state):
        if random.random() < self.eval_epsilon:
            return self.choose_random_action()
        else:
            return self.choose_greedy_action(state)

    def choose_action(self, state, step_num):
        self.display_from_action(step_num)
        if self.training and step_num < self.hyperparams['updates_per_iter']:
            # self.update_epsilon()
            return self.choose_random_action()
        elif self.training:
            return self.choose_train_action(state)
        else:
            return self.choose_eval_action(state)

    def have_experience(self, experience):
        if self.training:
            # state, action, reward, next_state = experience
            self.experience_replay.add_experience(experience)

    def batch_preprocessor(self, batch):
        batch = batch['sample']
        reward_discount = self.hyperparams['reward_discount']

        train_xs = {'state': np.array([s for (s, a, r, n) in batch])}
        train_y = np.zeros((len(batch),))
        train_mask = np.zeros((len(batch), len(self.actions)))
        preds = self.q_model.compute_target_preds(
            {'state': np.array([n for (s, a, r, n) in batch])},
            sess=self.sess)
        for experience_num, (state, action, reward, next_state) in \
                enumerate(batch):
            exp_returns = np.max(preds[experience_num])
            target = reward + reward_discount * exp_returns
            train_y[experience_num] = target
            train_mask[experience_num, np.argmax(action)] = 1
        self.target = target
        self.exp_returns = exp_returns
        self.mean_state_val = np.mean(np.array(
            [s for (s, a, r, n) in batch]))
        self.mean_preds = np.mean(preds)
        return train_xs, train_y, train_mask

    def make_feed_dict(self, xs, y, mask, train=True):
        feed_dict = create_x_feed_dict(self.q_model.get_input_vars(), xs)
        feed_dict.update(create_y_feed_dict(self.y, y))
        feed_dict.update({self.mask: mask})
        if train:
            feed_dict.update(create_supp_train_feed_dict(self.q_model))
        else:
            feed_dict.update(create_supp_test_feed_dict(self.q_model))
        return feed_dict

    def train_model(self,
                    train_xs,
                    train_y,
                    train_mask,
                    display=False):
        feed_dict = self.make_feed_dict(train_xs, train_y, train_mask,
                                        train=True)
        self.sess.run(self.train_step,
                      feed_dict=feed_dict)
        if display:
            loss = self.loss.loss.eval(
                feed_dict=feed_dict, session=self.sess)
            print('Loss: %f' % loss)
            print('Target: %f' % self.target)
            print('Exp returns: %f' % self.exp_returns)
            print('Mean state val: %f' % self.mean_state_val)
            print('Mean preds: %f' % self.mean_preds)
            if self.greedy_ind is not None:
                print('Greedy ind: %d' % self.greedy_ind)
                self.greedy_ind = None

    def train_net(self):
        batch_size = self.hyperparams['batch_size']
        sample = self.experience_replay.sample(
            len(self.experience_replay))
        minibatches = create_minibatch_iterator({'sample': sample}, None,
                                                self.batch_preprocessor,
                                                batch_size)
        num_updates = self.hyperparams['updates_per_iter']
        epoch = 0
        print('-' * 40 + 'Epoch %d' % epoch + '-' * 40)
        for update_num in range(num_updates):
            try:
                batch = next(minibatches)
            except StopIteration:
                minibatches = create_minibatch_iterator(
                    {'sample': sample}, None,
                    self.batch_preprocessor, batch_size)
                batch = next(minibatches)
                epoch += 1
                print('-' * 40 + 'Epoch %d' % epoch + '-' * 40)
            display = update_num % self.hyperparams['display_freq'] == 0
            self.train_model(batch[0], batch[1], batch[2], display)
        self.experience_replay.clear()

    def start_eval_mode(self):
        self.training = False

    def end_eval_mode(self):
        self.training = True

    def display_train_update(self, step_num):
        print('Step %d' % step_num)
        print('Epsilon: %f' % self.epsilon)
        # print('fc_l7_w: %s' % str(self.sess.run(self.fc_l7_w)))
        if self.target:
            print('Target: %f' % self.target)
        if self.exp_returns:
            print('Exp returns: %f' % self.exp_returns)
        if self.zero_pred_returns:
            print('Zero pred returns: %f' % self.zero_pred_returns)
        if self.exp_pred_returns:
            print('Exp pred returns: %f' % self.exp_pred_returns)
        if len(self.recent_train_q) > 0:
            print('Avg recent pred q: %f' %
                  (sum(self.recent_train_q) / len(self.recent_train_q)))
        if hasattr(self, 'preds'):
            print('Preds: %s' % str(self.preds))

    def display_eval_update(self, step_num):
        print('Step %d' % step_num)
        print('Epsilon: %f' % self.eval_epsilon)
        if self.greedy_ind is not None:
            print('Greedy ind: %d' % self.greedy_ind)
            self.greedy_ind = None
        if len(self.recent_eval_q) > 0:
            print('Avg recent pred q: %f' %
                  (sum(self.recent_eval_q) / len(self.recent_eval_q)))
            print('Current pred q: %f' % self.recent_eval_q[-1])
        if hasattr(self, 'preds'):
            print('Preds: %s' % str(self.preds))

    def learn(self, step_num):
        if (step_num + 1) % self.hyperparams['frames_per_epoch'] == 0:
            if len(self.experience_replay) >= \
               self.hyperparams['batch_size']:
                self.train_net()
                self.update_target_weights()
