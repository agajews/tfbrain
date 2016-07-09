from tfbrain.helpers import bcolors


class RLTrainer(object):
    def __init__(self, hyperparams, agent, task):
        self.hyperparams = hyperparams
        self.agent = agent
        self.task = task
        self.init_agent()

    def init_agent(self):
        self.agent.set_actions(self.task.get_actions())
        self.agent.set_state_shape(self.task.get_state_shape())
        self.agent.build()

#     def display_episode_start(self):
#         print('=' * 47 + ' Episode %d Start ' % self.episode_num + '=' * 47)

#     def display_episode(self, step_num):
#         print('=' * 47 + ' Episode %d Stats ' % self.episode_num + '=' * 47)
#         print('Frame num: %d' % step_num)
#         print('Total reward: %f' % self.recent_train_rewards[-1])
#         print('Avg recent reward: %f' %
#               (sum(self.recent_train_rewards) /
#                len(self.recent_train_rewards)))

#     def display_eval_start(self):
#         print('=' * 48 + ' Eval %d start ' % self.eval_num + '=' * 48)

#     def display_eval(self):
#         print('=' * 48 + ' Eval %d Stats ' % self.eval_num + '=' * 48)
#         print('Total reward: %d' % self.recent_eval_rewards[-1])
#         print('Avg recent reward: %f' %
#               (sum(self.recent_eval_rewards) /
#                len(self.recent_eval_rewards)))

#     def prep_eval(self):
#         num_recent = int(self.hyperparams['num_recent_episodes'] /
#                          self.hyperparams['eval_freq'])
#         self.recent_eval_rewards = deque(
#             maxlen=num_recent)
#         self.eval_num = 0

#     def eval(self):
#         self.display_eval_start()
#         self.agent.start_eval_mode()
#         self.task.start_episode()
#         total_reward = 0
#         step_num = 0
#         while not self.task.episode_is_over():
#             state = self.task.get_state()
#             action = self.agent.choose_action(state, step_num)
#             self.task.perform_action(action)
#             reward = self.task.get_reward()
#             total_reward += reward
#             step_num += 1
#         self.recent_eval_rewards.append(total_reward)
#         self.agent.end_eval_mode()
#         self.display_eval()
#         self.eval_num += 1

#     def prep_train(self):
#         self.recent_train_rewards = deque(
#             maxlen=self.hyperparams['num_recent_episodes'])
#         self.episode_num = 0

    # def perform_action(self, step_num):
        # self.agent.learn(step_num)
        # return reward

    # def train_episode(self):
    #     self.agent.learn(self.episode_num)

    # def train(self):
    #     self.prep_train()
    #     self.prep_eval()
    #     self.task.start_episode()
    #     total_reward = 0
    #     self.display_episode_start()
    #     for step_num in range(self.hyperparams['num_frames']):
    #         # don't save first iteration
    #         if (step_num + 1) % self.hyperparams['save_freq'] == 0:
    #             print('Saving params ...')
    #             self.agent.save_params()
    #         if self.task.episode_is_over():
    #             if self.episode_num % self.hyperparams['update_freq'] == 0:
    #                 self.train_episode()
    #             if self.episode_num % self.hyperparams['eval_freq'] == 0:
    #                 self.eval()
    #             self.recent_train_rewards.append(total_reward)
    #             self.display_episode(step_num)
    #             total_reward = 0
    #             self.episode_num += 1
    #             self.task.start_episode()
    #             self.display_episode_start()
    #         total_reward += self.perform_action(step_num)

    # def learn(self, step_num):
    #     self.agent.learn(step_num)

    def run_epoch(self,
                  epoch_num,
                  episode_num,
                  step_start,
                  num_frames,
                  train=True):
        start_episode_num = episode_num
        episode_step_num = 0
        if train:
            epoch_text = 'Train'
        else:
            epoch_text = 'Eval'
            self.agent.start_eval_mode()
        print(bcolors.OKGREEN +
              '-' * 40 + '%s Epoch %d' % (epoch_text, epoch_num) + '-' * 40 +
              bcolors.ENDC)
        total_reward = 0
        epoch_rewards = []
        self.task.start_episode()
        for step_num in range(step_start, step_start + num_frames):
            episode_step_num += 1
            if self.task.episode_is_over():
                epoch_rewards.append(total_reward)
                # print(bcolors.OKBLUE +
                #       '[[[ Episode %d finished with reward %d ]]]' %
                #       (episode_num, total_reward) +
                #       bcolors.ENDC)
                print(bcolors.OKBLUE +
                      '[[[ Episode %d finished with reward %d ]]]' %
                      (episode_num, self.task.get_episode_reward()) +
                      bcolors.ENDC)
                total_reward = 0
                episode_step_num = 0
                episode_num += 1
                self.task.start_episode()
            state = self.task.get_state()
            action = self.agent.choose_action(state, step_num)
            self.task.perform_action(action)
            reward = self.task.get_reward()
            next_state = self.task.get_state()
            experience = (state, action, reward, next_state)
            if train:
                self.agent.have_experience(experience)
                if episode_step_num % self.hyperparams['update_freq'] == 0:
                    self.agent.learn(step_num)
        if not train:
            self.agent.end_eval_mode()
        print(bcolors.OKGREEN +
              '%s %d had average reward %f' %
              (epoch_text, epoch_num,
               sum(epoch_rewards) / len(epoch_rewards)) +
              bcolors.ENDC)
        return episode_num - start_episode_num

    def train_by_epoch(self):
        episode_num = 0
        step_num = 0
        for epoch_num in range(self.hyperparams['num_epochs']):
            episode_num += self.run_epoch(epoch_num,
                                          episode_num,
                                          step_num,
                                          self.hyperparams['frames_per_epoch'],
                                          train=True)
            step_num += self.hyperparams['frames_per_epoch']
            episode_num += self.run_epoch(epoch_num,
                                          episode_num,
                                          step_num,
                                          self.hyperparams['frames_per_eval'],
                                          train=False)
            step_num += self.hyperparams['frames_per_eval']
            print('Saving params ...')
            self.agent.save_params()
