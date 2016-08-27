from collections import deque

from tfbrain.helpers import create_x_feed_dict, \
    create_y_feed_dict, create_supp_train_feed_dict, \
    create_supp_test_feed_dict, softmax, create_minibatch_iterator, \
    get_input_vars, get_output

import tensorflow as tf

import random

import numpy as np

# from memory_profiler import profile

from tfbrain.helpers import get_all_params, \
    set_all_params_ops
from tfbrain.memory import SNDQNExperienceReplay


class AsyncEdistActionChooser(object):
    def __init__(self, hyperparams, q_model, sess, actions):
        self.hyperparams = hyperparams
        self.sess = sess
        with tf.device('/cpu:0'):
            self.net = q_model.create_net(trainable=False)
        self.sess.run(tf.initialize_all_variables())
        self.input_vars = get_input_vars(self.net)
        self.y_hat = get_output(self.net)
        self.experience_cache = []
        self.episode_cache = []
        self.recent_train_q = deque(
            maxlen=self.hyperparams['num_recent_steps'])
        self.recent_eval_q = deque(
            maxlen=self.hyperparams['num_recent_steps'])
        self.prepare_epsilon()
        self.training = True
        self.actions = actions
        self.greedy_ind = None

    def compute_preds(self, xs):
        feed_dict = create_x_feed_dict(self.input_vars, xs)
        feed_dict.update(create_supp_test_feed_dict(self))
        preds = self.y_hat.eval(feed_dict=feed_dict,
                                session=self.sess)
        return preds

    def prepare_epsilon(self):
        self.epsilons = []
        self.epsilon_steps = []
        self.epsilon_ends = []
        self.epsilon_probs = []
        for start, end, anneal_len, prob in self.hyperparams['epsilon']:
            self.epsilons.append(start)
            self.epsilon_steps.append(self.hyperparams['num_threads'] *
                                      (end - start) / anneal_len)
            self.epsilon_ends.append(end)
            self.epsilon_probs.append(prob)
        # epsilon = self.hyperparams['epsilon']
        # self.epsilon = epsilon[0]
        # self.epsilon_step = (self.hyperparams['num_threads'] *
        #                      (epsilon[1] - epsilon[0]) / epsilon[2])
        self.eval_epsilon = self.hyperparams['eval_epsilon']

    def start_episode(self):
        if self.training:
            self.episode_cache = []
        self.epsilon_ind = np.random.choice(len(self.epsilons),
                                            p=self.epsilon_probs)
        self.epsilon = self.epsilons[self.epsilon_ind]

    def end_episode(self):
        if self.training:
            self.experience_cache.append(self.episode_cache)

    def start_eval_mode(self):
        self.training = False

    def end_eval_mode(self):
        self.training = True

    def get_single_state_dict(self, single_state):
        xs = {'state': np.reshape(single_state, (1,) + single_state.shape)}
        return xs

    def compute_single_state_preds(self, single_state):
        xs = self.get_single_state_dict(single_state)
        preds = self.compute_preds(xs)
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
        # if self.epsilon > self.hyperparams['epsilon'][1]:
        #     self.epsilon += self.epsilon_step
        for epsilon_num, epsilon in enumerate(self.epsilons):
            if epsilon > self.epsilon_ends[epsilon_num]:
                self.epsilons[epsilon_num] += self.epsilon_steps[epsilon_num]
        self.epsilon = self.epsilons[self.epsilon_ind]

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
        if self.training:
            return self.choose_train_action(state)
        else:
            return self.choose_eval_action(state)

    def have_experience(self, experience):
        if self.training:
            # state, action, reward, next_state = experience
            # self.experience_replay.add_experience(experience)
            self.episode_cache.append(experience)

    def display_train_update(self, step_num):
        print('Step %d' % step_num)
        print('Epsilon: %f; Ind: %d' % (self.epsilon, self.epsilon_ind))
        # print('fc_l7_w: %s' % str(self.sess.run(self.fc_l7_w)))
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


class AsyncEdistSNDQNAgent(object):
    def __init__(self, hyperparams, q_model, optim, loss, param_fnm,
                 dtype=np.uint8):
        self.hyperparams = hyperparams
        self.q_model = q_model
        self.optim = optim
        self.loss = loss
        self.param_fnm = param_fnm
        self.dtype = dtype

    def set_actions(self, actions):
        self.actions = actions
        self.q_model.set_num_actions(len(actions))

    def set_state_shape(self, state_shape):
        self.state_shape = state_shape
        self.q_model.set_state_shape(state_shape)

    def prepare_debug_vars(self):
        self.target = None
        self.exp_returns = None
        self.greedy_ind = None
        self.exp_pred_returns = None
        self.zero_pred_returns = None

    def build(self):
        print('Building agent ...')
        self.tmax = self.hyperparams['tmax']
        with tf.device('/gpu:0'):
            self.q_model.setup_net()
        self.mask = tf.placeholder(shape=(None, len(self.actions)),
                                   dtype=tf.float32,
                                   name='mask')
        self.y = tf.placeholder(
            dtype=tf.float32,
            shape=(None,),
            name='expected_y')
        self.loss.build(self.q_model.y_hat, self.y, self.mask)
        self.learning_rate_var = tf.placeholder(tf.float32)
        self.train_step = self.optim.get_train_step(self.loss.loss,
                                                    self.learning_rate_var)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        # self.sess.run(tf.initialize_variables(flatten_params(
        #     get_all_params(self.q_model.net))))

        self.prepare_debug_vars()
        self.training = True
        self.experience_replay = SNDQNExperienceReplay(self.hyperparams,
                                                       self.dtype)
        self.experience_replay.build()

        self.update_target_weights_ops = set_all_params_ops(
            get_all_params(self.q_model.get_target_net()),
            get_all_params(self.q_model.get_net()))
        self.update_target_weights()
        self.update_chooser_weights_ops = []
        self.prepare_learning_rate()

    def prepare_learning_rate(self):
        self.learning_rate = self.hyperparams['learning_rate'][0]
        self.learning_rate_step = ((self.hyperparams['learning_rate'][1] -
                                    self.hyperparams['learning_rate'][0]) /
                                   self.hyperparams['learning_rate'][2] *
                                   self.hyperparams['frames_per_epoch'])

    def update_learning_rate(self):
        self.learning_rate += self.learning_rate_step

    def build_action_chooser(self):
        chooser = AsyncEdistActionChooser(self.hyperparams, self.q_model,
                                          self.sess, self.actions)
        self.update_chooser_weights_ops += set_all_params_ops(
            get_all_params(chooser.net),
            get_all_params(self.q_model.get_net()))
        return chooser

    def update_chooser_weights(self):
        print('Updating chooser weights ...')
        self.sess.run(self.update_chooser_weights_ops)

    def aggregate_from(self, chooser):
        reward_discount = self.hyperparams['reward_discount']
        for episode_cache in chooser.experience_cache:
            for batch_num in range(
                    0, len(episode_cache), self.tmax):
                batch = episode_cache[batch_num:batch_num + self.tmax]
                value = 0
                last_state = batch[-1][3]
                for experience_num, experience in enumerate(reversed(batch)):
                    state, action, reward, next_state = experience
                    experience = (state, action, reward, last_state, value)
                    self.experience_replay.add_experience(experience)
                    value += reward_discount ** experience_num * reward
        chooser.experience_cache = []

    def save_params(self):
        self.q_model.save_params(self.param_fnm,
                                 sess=self.sess)

    def load_params(self):
        self.q_model.load_params(self.param_fnm,
                                 sess=self.sess)
        self.update_target_weights()

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

    def compute_single_state_target_preds(self, single_state):
        xs = self.get_single_state_dict(single_state)
        preds = self.q_model.compute_target_preds(xs, sess=self.sess)
        return preds

    def batch_preprocessor(self, batch):
        batch = batch['sample']
        reward_discount = self.hyperparams['reward_discount']
        tmax = self.hyperparams['tmax']

        train_xs = {'state': np.array([s for (s, a, r, n, v) in batch])}
        train_y = np.zeros((len(batch),))
        train_mask = np.zeros((len(batch), len(self.actions)))
        preds = self.q_model.compute_target_preds(
            {'state': np.array([n for (s, a, r, n, v) in batch])},
            sess=self.sess)
        for experience_num, (state, action, reward, next_state, value) in \
                enumerate(batch):
            exp_returns = np.max(preds[experience_num])
            target = (reward +
                      (reward_discount * value) +
                      ((reward_discount ** tmax) * exp_returns))
            train_y[experience_num] = target
            train_mask[experience_num, np.argmax(action)] = 1
        self.target = target
        self.exp_returns = exp_returns
        # self.mean_state_val = np.mean(np.array(
        #     [s for (s, a, r, n) in batch]))
        # self.mean_preds = np.mean(preds)
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
        feed_dict.update({self.learning_rate_var: self.learning_rate})
        self.sess.run(self.train_step,
                      feed_dict=feed_dict)
        if display:
            loss = self.loss.loss.eval(
                feed_dict=feed_dict, session=self.sess)
            print('Learning rate: %f' % self.learning_rate)
            print('Loss: %f' % loss)
            print('Target: %f' % self.target)
            print('Exp returns: %f' % self.exp_returns)
            # print('Mean state val: %f' % self.mean_state_val)
            # print('Mean preds: %f' % self.mean_preds)
            if self.greedy_ind is not None:
                print('Greedy ind: %d' % self.greedy_ind)
                self.greedy_ind = None

    # @profile
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
            if display:
                print('Update num: %d' % update_num)
            self.train_model(batch[0], batch[1], batch[2], display)
        self.experience_replay.clear()
        self.update_learning_rate()

    def learn(self):
        if len(self.experience_replay) >= \
           self.hyperparams['batch_size']:
            self.train_net()
            self.update_target_weights()
