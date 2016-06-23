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

    def display_episode_start(self, episode_num, training):
        print('=' * 47 + ' Episode %d Start ' % episode_num + '=' * 47)
        if training:
            print('TRAINING')
        else:
            print('EVALUATING')

    def display_episode(self, episode_num,
                        frame_num,
                        episode_experiences,
                        total_reward,
                        recent_total_rewards,
                        training):
        print('=' * 47 + ' Episode %d Stats ' % episode_num + '=' * 47)
        print('Frame num: %d' % frame_num)
        print('Total reward: %f' % total_reward)
        print('Avg recent reward: %f' %
              (sum(recent_total_rewards) / len(recent_total_rewards)))
        self.display_episode_start(episode_num + 1, training)

    def train(self):
        recent_total_rewards = deque(
            maxlen=self.hyperparams['num_recent_rewards'])
        self.task.start_episode()
        total_reward = 0
        episode_num = 0
        episode_experiences = []
        self.agent.start_eval_mode()
        training = False
        self.display_episode_start(episode_num, training)
        for frame_num in range(self.hyperparams['num_frames']):
            if frame_num % self.hyperparams['save_freq'] == 0:
                print('Saving params ...')
                self.agent.save_params()
            if self.task.episode_is_over():
                if not training:
                    self.agent.end_eval_mode()
                    training = True
                if (episode_num + 1) % self.hyperparams['eval_freq'] == 0:
                    self.agent.start_eval_mode()
                    training = False
                self.task.start_episode()
                recent_total_rewards.append(total_reward)
                self.display_episode(episode_num,
                                     frame_num,
                                     episode_experiences,
                                     total_reward,
                                     recent_total_rewards,
                                     training)
                total_reward = 0
                episode_num += 1
                episode_experiences = []
            state = self.task.get_state()
            action = self.agent.choose_action(state)
            self.task.perform_action(action)
            reward = self.task.get_reward()
            total_reward += reward
            next_state = self.task.get_state()
            experience = (state, action, reward, next_state)
            episode_experiences.append(experience)
            self.agent.have_experience(experience)
            state = next_state
            self.agent.learn(frame_num)
