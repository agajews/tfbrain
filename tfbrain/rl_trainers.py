from collections import deque


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

    def display_episode_start(self):
        print('=' * 47 + ' Episode %d Start ' % self.episode_num + '=' * 47)

    def display_episode(self, frame_num):
        print('=' * 47 + ' Episode %d Stats ' % self.episode_num + '=' * 47)
        print('Frame num: %d' % frame_num)
        print('Total reward: %f' % self.recent_train_rewards[-1])
        print('Avg recent reward: %f' %
              (sum(self.recent_train_rewards) /
               len(self.recent_train_rewards)))

    def display_eval_start(self):
        print('=' * 48 + ' Eval %d start ' % self.eval_num + '=' * 48)

    def display_eval(self):
        print('=' * 48 + ' Eval %d Stats ' % self.eval_num + '=' * 48)
        print('Total reward: %d' % self.recent_eval_rewards[-1])
        print('Avg recent reward: %f' %
              (sum(self.recent_eval_rewards) /
               len(self.recent_eval_rewards)))

    def prep_eval(self):
        self.recent_eval_rewards = deque(
            maxlen=self.hyperparams['num_recent_rewards'])
        self.eval_num = 0

    def eval(self):
        self.display_eval_start()
        self.agent.start_eval_mode()
        self.task.start_episode()
        total_reward = 0
        frame_num = 0
        while not self.task.episode_is_over():
            state = self.task.get_state()
            action = self.agent.choose_action(state, frame_num)
            self.task.perform_action(action)
            reward = self.task.get_reward()
            total_reward += reward
            frame_num += 1
        self.recent_eval_rewards.append(total_reward)
        self.agent.end_eval_mode()
        self.display_eval()
        self.eval_num += 1

    def prep_train(self):
        self.recent_train_rewards = deque(
            maxlen=self.hyperparams['num_recent_rewards'])
        self.episode_num = 0

    def train_frame(self, frame_num):
        state = self.task.get_state()
        action = self.agent.choose_action(state, frame_num)
        self.task.perform_action(action)
        reward = self.task.get_reward()
        next_state = self.task.get_state()
        experience = (state, action, reward, next_state)
        self.agent.have_experience(experience)
        self.agent.learn(frame_num)
        return reward

    def train(self):
        self.prep_train()
        self.prep_eval()
        self.task.start_episode()
        total_reward = 0
        self.display_episode_start()
        for frame_num in range(self.hyperparams['num_frames']):
            if frame_num % self.hyperparams['save_freq'] == 0:
                print('Saving params ...')
                self.agent.save_params()
            if self.task.episode_is_over():
                if self.episode_num % self.hyperparams['eval_freq'] == 0:
                    self.eval()
                self.recent_train_rewards.append(total_reward)
                self.display_episode(frame_num)
                total_reward = 0
                self.episode_num += 1
                self.task.start_episode()
                self.display_episode_start()
            total_reward += self.train_frame(frame_num)
