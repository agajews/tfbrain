from tfbrain.helpers import create_x_feed_dict, \
    create_y_feed_dict, create_supp_train_feed_dict, \
    create_supp_test_feed_dict, softmax

from tfbrain.memory import RDRLMem

import tensorflow as tf

import random

import numpy as np

from tfbrain.helpers import get_all_params, \
    get_output, get_input_hidden_vars, get_output_hidden_vars, \
    flatten_params, get_init_hidden, set_all_params_ops, \
    get_trainable_params, create_minibatch_iterator


class RDRLAgent(object):
    def __init__(self, hyperparams,
                 action_model, action_optim,
                 state_model, state_loss, state_optim,
                 reward_model, reward_loss, reward_optim,
                 value_model, value_loss, value_optim,
                 param_fnm):
        self.hyperparams = hyperparams
        self.param_fnm = param_fnm

        self.action_model = action_model
        self.action_optim = action_optim

        self.state_model = state_model
        self.state_loss = state_loss
        self.state_optim = state_optim

        self.reward_model = reward_model
        self.reward_loss = reward_loss
        self.reward_optim = reward_optim

        self.value_model = value_model
        self.value_loss = value_loss
        self.value_optim = value_optim

    def start_episode(self):
        value = 0
        for part_experience in reversed(self.part_experiences):
            state, action, reward, next_state = part_experience
            experience = (state, action, reward, next_state, value)
            self.experience_replay.add_experience(experience)
            value += reward
        self.part_experiences = []

        # zero hidden states
        feed_dict = create_x_feed_dict(
            self.action_model.get_input_vars(),
            {'state': np.zeros((1, 1) + self.state_shape)})
        for hidden_name, hidden_state in self.hidden_states.items():
            self.hidden_state_vals[hidden_state] = np.zeros(
                self.init_hidden[hidden_name].eval(
                    session=self.sess,
                    feed_dict=feed_dict).shape)

    def set_actions(self, actions):
        self.actions = actions
        for model in [self.action_model,
                      self.state_model,
                      self.reward_model,
                      self.value_model]:
            model.set_num_actions(len(actions))

    def set_state_shape(self, state_shape):
        self.state_shape = state_shape
        for model in [self.action_model,
                      self.state_model,
                      self.reward_model,
                      self.value_model]:
            model.set_state_shape(state_shape)

    def prepare_epsilon(self):
        epsilon = self.hyperparams['epsilon']
        self.epsilon = epsilon[0]
        self.epsilon_step = (epsilon[1] - epsilon[0]) / epsilon[2]
        self.eval_epsilon = self.hyperparams['eval_epsilon']

    def build(self):
        print('Building agent ...')
        self.action_model.setup_net()

        self.state_model.setup_net()
        self.state_y = tf.placeholder(
            dtype=tf.float32,
            shape=(None, np.prod(self.state_shape)),
            name='expected_state_y')
        self.state_loss.build(self.state_model.y_hat, self.state_y)
        state_params = flatten_params(get_all_params(
            self.state_model.get_net()))
        # print(state_params)
        # self.state_train_step = self.state_optim.get_train_step(
        #     self.state_loss.loss, state_params)

        self.reward_model.setup_net()
        self.reward_y = tf.placeholder(
            dtype=tf.float32,
            shape=(None,),
            name='expected_reward_y')
        self.reward_loss.build(self.reward_model.y_hat, self.reward_y)
        reward_params = flatten_params(get_all_params(
            self.reward_model.get_net()))
        # self.reward_train_step = self.reward_optim.get_train_step(
        #     self.reward_loss.loss, reward_params)

        self.value_model.setup_net()
        self.value_y = tf.placeholder(
            dtype=tf.float32,
            shape=(None,),
            name='expected_value_y')
        self.value_loss.build(self.value_model.y_hat, self.value_y)
        value_params = flatten_params(get_all_params(
            self.value_model.get_net()))
        # self.value_train_step = self.value_optim.get_train_step(
        #     self.value_loss.loss, value_params)

        partial_params = state_params + reward_params + value_params
        partial_loss = (self.state_loss.loss +
                        self.reward_loss.loss +
                        self.value_loss.loss)
        self.partial_train_step = self.state_optim.get_train_step(
            partial_loss, partial_params)

        reward_discount = self.hyperparams['reward_discount']
        batch_size = self.hyperparams['batch_size']
        self.seed_train_state = tf.placeholder(
            tf.float32,
            shape=(batch_size,) + self.state_shape,
            name='seed_train_state')

        # scale = self.hyperparams['action_train_scale']
        value_rollout_length = self.hyperparams['value_rollout_length']
        next_state = self.seed_train_state
        next_conv_state = tf.concat(3, [next_state] * value_rollout_length)
        total_reward = tf.zeros((batch_size,))
        for timestep in range(self.hyperparams['rollout_length']):
            state = next_state
            conv_state = next_conv_state

            action = get_output(self.action_model.get_net(),
                                {'state': tf.expand_dims(state, 1)},
                                timestep=True)

            # evil softmax to closer-to-one-hot magic
            # action_max = tf.reduce_max(action, reduction_indices=1)
            # action_max = tf.expand_dims(action_max, 1)

            # action_min = tf.reduce_min(action, reduction_indices=1)
            # action_min = tf.expand_dims(action_min, 1)

            # action = tf.pow((1 - (action_max - action) -
            #                  (1 - (action_max - action_min))) /
            #                 (action_max - action_min), scale)
            # print('action shape')
            # print(action.get_shape())

            next_state = get_output(self.state_model.get_net(),
                                    {'state': conv_state,
                                     'action': action})
            next_state = tf.reshape(next_state, (-1, *self.state_shape))
            next_conv_state = tf.concat(
                3, [next_conv_state[:, :, :, :value_rollout_length - 1],
                    next_state])

            reward = get_output(self.reward_model.net,
                                {'state': next_conv_state,
                                 'action': action})
            total_reward += reward_discount * tf.squeeze(reward, [1])
            value = get_output(self.value_model.get_net(),
                               {'state': next_conv_state})

        print('reward shape')
        print(reward.get_shape())
        print('value shape')
        print(value.get_shape())
        total_reward += reward_discount * tf.squeeze(value, [1])
        print('Total reward shape')
        print(total_reward.get_shape())
        self.exp_returns = tf.reduce_mean(total_reward)

        print('Flattening params ...')
        action_params = flatten_params(get_trainable_params(
            self.action_model.get_net()))
        print('Action params:')
        print(get_trainable_params(self.action_model.get_net()))
        self.action_train_step = self.action_optim.get_train_step(
            -self.exp_returns, action_params)

        self.action_preds = get_output(self.action_model.get_net(),
                                       None,
                                       timestep=True,
                                       input_hidden=True)
        # self.assign_hidden_ops = get_assign_hidden_ops(
        #     self.action_model.get_net())
        # self.zero_hidden_ops = get_assign_hidden_ops(
        #     self.action_model.get_net(),
        #     zero=True)
        # self.hidden_states = get_input_hidden_vars(
        #     self.action_model.get_net(),
        #     timestep=True)

        self.hidden_states = get_input_hidden_vars(
            self.action_model.get_net(),
            timestep=True)
        self.hidden_output_states = get_output_hidden_vars(
            self.action_model.get_net())
        self.hidden_state_vals = {}
        self.init_hidden = get_init_hidden(
            self.action_model.get_net())
        # for hidden_name, hidden_state in self.hidden_states.items():
        #     self.hidden_state_vals[hidden_state] = np.zeros(
        #         hidden_state.eval(session=self.sess).shape)
        #     self.hidden_state_vals[hidden_state] = None

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        self.update_value_target_weights_ops = set_all_params_ops(
            get_all_params(self.value_model.get_target_net()),
            get_all_params(self.value_model.get_net()))
        self.update_value_target_weights()

        self.prepare_epsilon()
        self.training = True
        self.part_experiences = []

        self.experience_replay = RDRLMem(self.hyperparams)
        self.experience_replay.build()

        self.greedy_ind = None

    def update_value_target_weights(self):
        print('Updating target weights ...')
        self.sess.run(self.update_value_target_weights_ops[0:])

    def get_single_state_dict(self, single_state):
        xs = {'state': np.reshape(single_state, (1, 1) + single_state.shape)}
        return xs

    def get_single_state_feed_dict(self, single_state):
        xs = self.get_single_state_dict(single_state)
        feed_dict = create_x_feed_dict(
            self.action_model.get_input_vars(), xs)
        feed_dict.update(self.hidden_state_vals)
        feed_dict.update(create_supp_test_feed_dict(self.action_model))
        return feed_dict

    def update_action_hidden(self, single_state):
        feed_dict = self.get_single_state_feed_dict(single_state)
        for hidden_name, hidden_state in self.hidden_states.items():
            self.hidden_state_vals[hidden_state] = \
                self.hidden_output_states[hidden_name].eval(
                    session=self.sess, feed_dict=feed_dict)

    def compute_single_state_preds(self, single_state):
        # print([s.eval(session=self.sess)
        #        for (k, s) in self.hidden_states.items()])
        # self.sess.run([i for (k, i) in self.assign_hidden_ops.items()],
        #               feed_dict=feed_dict)
        # self.update_action_hidden(single_state)
        feed_dict = self.get_single_state_feed_dict(single_state)
        preds = self.action_model.y_hat.eval(feed_dict=feed_dict,
                                             session=self.sess)
        return preds

    def choose_greedy_action(self, state):
        preds = self.compute_single_state_preds(state)
        self.preds = preds
        greedy_ind = np.argmax(preds)
        self.greedy_ind = greedy_ind
        return preds

    def display_from_action(self, step_num):
        if step_num % self.hyperparams['display_freq'] == 0:
            print('Step %d' % step_num)
            if self.training:
                print('Epsilon: %f' % self.epsilon)
            if self.greedy_ind is not None:
                print('Greedy ind: %d' % self.greedy_ind)
                self.greedy_ind = None
            if hasattr(self, 'preds'):
                print('Preds: %s' % str(self.preds))
            # if self.training:
            #     self.display_train_update(step_num)
            # else:
            #     self.display_eval_update(step_num)

    def choose_random_action(self):
        return softmax(np.expand_dims(np.random.randn(len(self.actions)), 0))

    def update_epsilon(self):
        if self.epsilon > self.hyperparams['epsilon'][1]:
            self.epsilon += self.epsilon_step

    def choose_train_action(self, state, step_num):
        if step_num > self.hyperparams['init_explore_len']:
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
        self.update_action_hidden(state)
        self.display_from_action(step_num)
        if self.training and step_num < self.hyperparams['init_explore_len']:
            return self.choose_random_action()
        elif self.training:
            return self.choose_train_action(state, step_num)
        else:
            return self.choose_eval_action(state)

    def have_experience(self, part_experience):
        if self.training:
            # state, action, reward, next_state = part_experience
            self.part_experiences.append(part_experience)

    def batch_preprocessor(self, batch):
        new_batch = []
        for experience_num, experience in enumerate(batch['sample']):
            new_experience = []
            for frame_num, frame_experience in enumerate(experience):
                state, action, reward, next_state, value = frame_experience
                state = self.experience_replay.decompress(state)
                next_state = self.experience_replay.decompress(next_state)
                frame_experience = (state, action, reward, next_state, value)
                new_experience.append(frame_experience)
            new_batch.append(new_experience)
        xs = {}
        ys = {}
        xs['state'], ys['state'] = self.preprocess_state_batch(new_batch)
        xs['reward'], ys['reward'] = self.preprocess_reward_batch(
            new_batch)
        xs['value'], ys['value'] = self.preprocess_value_batch(new_batch)
        return xs, ys

    def preprocess_state_batch(self, batch):
        rollout_length = self.hyperparams['value_rollout_length']
        xs = {'state': np.zeros((len(batch),) +
                                self.state_shape[:-1] + (rollout_length,)),
              'action': np.zeros((len(batch), len(self.actions)))}
        y = np.zeros((len(batch), np.prod(self.state_shape)))
        for experience_num, experience in enumerate(batch):
            for frame_num, frame_experience in enumerate(experience):
                state, action, reward, next_state, value = frame_experience
                xs['state'][experience_num, :, :, frame_num] = np.squeeze(
                    state, 2)
            xs['action'][experience_num] = action
            y[experience_num] = next_state.flatten()  # last next_state in seq
        return xs, y

    def gen_state_train_data(self):
        batch_size = self.hyperparams['batch_size']
        rollout_length = self.hyperparams['value_rollout_length']
        sample = self.experience_replay.sample(batch_size,
                                               rollout_length)
        return self.preprocess_state_batch(sample)

    def train_state_model(self):
        xs, y = self.gen_state_train_data()
        feed_dict = self.make_single_feed_dict(self.state_model, xs,
                                               self.state_y, y)
        self.sess.run(self.state_train_step,
                      feed_dict=feed_dict)
        loss = self.state_loss.loss.eval(
            feed_dict=feed_dict, session=self.sess)
        return loss

    def preprocess_reward_batch(self, batch):
        rollout_length = self.hyperparams['value_rollout_length']

        xs = {'state': np.zeros((len(batch),) +
                                self.state_shape[:-1] + (rollout_length,)),
              'action': np.zeros((len(batch),
                                  len(self.actions)))}
        y = np.zeros((len(batch),))
        for experience_num, experience in enumerate(batch):
            for frame_num, frame_experience in enumerate(experience):
                state, action, reward, next_state, value = frame_experience
                xs['state'][experience_num, :, :, frame_num] = np.squeeze(
                    state, 2)
            xs['action'][experience_num] = action  # last action in seq
            y[experience_num] = reward  # last reward in seq
        return xs, y

    def gen_reward_train_data(self):
        batch_size = self.hyperparams['batch_size']
        rollout_length = self.hyperparams['value_rollout_length']
        sample = self.experience_replay.sample(batch_size,
                                               rollout_length)
        return self.preprocess_reward_batch(sample)

    def train_reward_model(self):
        xs, y = self.gen_reward_train_data()
        feed_dict = self.make_single_feed_dict(self.reward_model, xs,
                                               self.reward_y, y)
        self.sess.run(self.reward_train_step,
                      feed_dict=feed_dict)
        loss = self.reward_loss.loss.eval(
            feed_dict=feed_dict, session=self.sess)
        return loss

    def preprocess_value_batch(self, batch):
        rollout_length = self.hyperparams['value_rollout_length']
        reward_discount = self.hyperparams['reward_discount']

        pred_xs = {'state': np.zeros(
            (len(batch),) + self.state_shape[:-1] + (rollout_length,))}

        for experience_num, experience in enumerate(batch):
            for frame_num, frame_experience in enumerate(experience):
                state, action, reward, next_state, value = frame_experience
                pred_xs['state'][experience_num, :, :, frame_num] = \
                    np.squeeze(state, 2)

        preds = self.value_model.compute_target_preds(pred_xs,
                                                      sess=self.sess)

        xs = {'state': np.zeros(
            (len(batch),) + self.state_shape[:-1] + (rollout_length,))}
        y = np.zeros((len(batch),))
        for experience_num, experience in enumerate(batch):
            for frame_num, frame_experience in enumerate(experience):
                state, action, reward, next_state, value = frame_experience
                xs['state'][experience_num, :, :, frame_num] = np.squeeze(
                    state, 2)
            y[experience_num] = (reward + reward_discount *
                                 preds[experience_num])
        self.mean_preds = np.mean(preds)
        return xs, y

    def gen_value_train_data(self):
        batch_size = self.hyperparams['batch_size']
        value_rollout_length = self.hyperparams['value_rollout_length']
        sample = self.experience_replay.sample(batch_size,
                                               value_rollout_length)
        return self.preprocess_value_batch(sample)

    # def train_value_model(self):
    #     xs, y = self.gen_value_train_data()
    #     feed_dict = self.make_feed_dict(self.value_model, xs,
    #                                     self.value_y, y)
    #     self.sess.run(self.value_train_step,
    #                   feed_dict=feed_dict)
    #     loss = self.value_loss.loss.eval(
    #         feed_dict=feed_dict, session=self.sess)
    #     return loss

    def gen_train_action_data(self):
        batch_size = self.hyperparams['batch_size']
        seed_state = np.zeros((batch_size,) + self.state_shape)
        sample = self.experience_replay.sample(batch_size, 1)
        for experience_num, experience in enumerate(sample):
            frame_experience = experience[0]
            state, action, reward, next_state, value = frame_experience
            seed_state[experience_num] = state
        return seed_state

    def train_action_model(self, num_updates, display=False, mega=False):
        if mega:
            print('-' * 40 + 'Training action model' + '-' * 40)
        for update in range(num_updates):
            seed_state = self.gen_train_action_data()
            feed_dict = {self.seed_train_state: seed_state}
            self.sess.run(self.action_train_step,
                          feed_dict=feed_dict)
            if display and update % self.hyperparams['display_freq'] == 0:
                if mega:
                    print('Train update %d' % update)
                exp_returns = self.exp_returns.eval(
                    feed_dict=feed_dict, session=self.sess)
                print('Exp returns: %f' % exp_returns)

    def make_single_feed_dict(self, model, xs, y_var, y_val, train=True):
        feed_dict = create_x_feed_dict(model.get_input_vars(), xs)
        feed_dict.update(create_y_feed_dict(y_var, y_val))
        if train:
            feed_dict.update(create_supp_train_feed_dict(model))
        else:
            feed_dict.update(create_supp_test_feed_dict(model))
        return feed_dict

    def make_feed_dict(self, models, y_vars, inputs, train=True):
        feed_dict = {}
        for model_name, model in models.items():
            feed_dict.update(create_x_feed_dict(model.get_input_vars(),
                                                inputs[0][model_name]))
            feed_dict.update(create_y_feed_dict(y_vars[model_name],
                                                inputs[1][model_name]))
        if train:
            for model_name, model in models.items():
                feed_dict.update(create_supp_train_feed_dict(model))
        else:
            for model_name, model in models.items():
                feed_dict.update(create_supp_test_feed_dict(model))
        return feed_dict

    def train_partial_model(self, num_updates,
                            mega=False, display=False):
        batch_size = self.hyperparams['batch_size']
        rollout_len = self.hyperparams['value_rollout_length']
        if mega:
            sample = self.experience_replay.sample(
                len(self.experience_replay) - rollout_len - 1, rollout_len,
                decompress=False)
        else:
            sample = self.experience_replay.sample(
                batch_size, rollout_len, decompress=False)
        minibatches = create_minibatch_iterator({'sample': sample}, None,
                                                self.batch_preprocessor,
                                                batch_size)
        epoch = 0
        if mega:
            print('-' * 40 + 'Epoch %d' % epoch + '-' * 40)
        for update in range(num_updates):
            try:
                batch = next(minibatches)
            except StopIteration:
                minibatches = create_minibatch_iterator(
                    {'sample': sample}, None,
                    self.batch_preprocessor, batch_size)
                batch = next(minibatches)
                epoch += 1
                if mega:
                    print('-' * 40 + 'Epoch %d' % epoch + '-' * 40)
            feed_dict = self.make_feed_dict({'state': self.state_model,
                                             'reward': self.reward_model,
                                             'value': self.value_model},
                                            {'state': self.state_y,
                                             'reward': self.reward_y,
                                             'value': self.value_y},
                                            batch)
            self.sess.run(self.partial_train_step,
                          feed_dict=feed_dict)
            if mega and update % self.hyperparams['update_target_freq'] == 0:
                self.update_value_target_weights()
            if display and update % self.hyperparams['display_freq'] == 0:
                if mega:
                    print('Partial update %d' % update)
                self.state_loss_val = self.state_loss.loss.eval(
                    feed_dict=feed_dict, session=self.sess)
                self.reward_loss_val = self.reward_loss.loss.eval(
                    feed_dict=feed_dict, session=self.sess)
                self.value_loss_val = self.value_loss.loss.eval(
                    feed_dict=feed_dict, session=self.sess)
                self.display_train_update(update)

    def train_model(self, display=False):
        if len(self.experience_replay) >= \
           self.hyperparams['batch_size']:
            self.train_partial_model(num_updates=1, display=display)
            self.train_action_model(num_updates=1, display=display)
            # self.value_loss_val = self.train_value_model()
            # self.state_loss_val = self.train_state_model()
            # self.reward_loss_val = self.train_reward_model()

    # def train_action(self):
    #     if len(self.experience_replay) >= \
    #        self.hyperparams['batch_size']:
    #         self.exp_returns_val = self.train_action_model()

    def start_eval_mode(self):
        self.training = False

    def end_eval_mode(self):
        self.training = True

    def display_train_update(self, step_num):
        # print('Step %d' % step_num)
        # print('Epsilon: %f' % self.epsilon)
        losses = ''
        if hasattr(self, 'state_loss_val'):
            losses += 'State loss: %f' % self.state_loss_val
            losses += ' | Reward loss: %f' % self.reward_loss_val
            losses += ' | Value loss: %f' % self.value_loss_val
            print(losses)
            print('Mean preds: %s' % str(self.mean_preds))
        # if hasattr(self, 'exp_returns_val'):
        #     print('Exp returns: %f' % self.exp_returns_val)
        # if hasattr(self, 'preds'):
        #     print('Preds: %s' % str(self.preds))
        # if self.greedy_ind is not None:
        #     print('Greedy ind: %d' % self.greedy_ind)
        #     self.greedy_ind = None

    def display_eval_update(self, step_num):
        print('Step %d' % step_num)
        print('Epsilon: %f' % self.eval_epsilon)
        if self.greedy_ind is not None:
            print('Greedy ind: %d' % self.greedy_ind)
            self.greedy_ind = None
        if hasattr(self, 'preds'):
            print('Preds: %s' % str(self.preds))

    def learn(self, step_num):
        # if step_num > self.hyperparams['init_explore_len']:
        #     if step_num % self.hyperparams['train_freq'] == 0:
        #         for i in range(self.hyperparams['updates_per_model_iter']):
        #             self.train_model()
        #     if step_num > self.hyperparams['init_model_train'] and \
        #        step_num % self.hyperparams['action_train_freq'] == 0:
        #         for i in range(self.hyperparams['updates_per_iter']):
        #             self.train_action()
        # if step_num % self.hyperparams['update_target_freq'] == 0:
        #     self.update_value_target_weights()
        if step_num == self.hyperparams['init_explore_len']:
            print('-' * 40 + 'Beginning mega' + '-' * 40)
            self.train_partial_model(self.hyperparams['num_mega_updates'],
                                     display=True, mega=True)
            self.train_action_model(self.hyperparams['num_mega_updates'],
                                    display=True, mega=True)
            print('-' * 40 + 'Resuming training' + '-' * 40)
        elif step_num > self.hyperparams['init_explore_len']:
            display = step_num % self.hyperparams['display_freq'] == 0
            self.train_model(display=display)
            if step_num % self.hyperparams['update_target_freq'] == 0:
                self.update_value_target_weights()

    def save_params(self):
        self.action_model.save_params(self.param_fnm + '.action',
                                      sess=self.sess)
        self.state_model.save_params(self.param_fnm + '.state',
                                     sess=self.sess)
        self.reward_model.save_params(self.param_fnm + '.reward',
                                      sess=self.sess)
        self.value_model.save_params(self.param_fnm + '.value',
                                     sess=self.sess)
