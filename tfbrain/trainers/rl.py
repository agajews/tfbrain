from tfbrain.helpers import bcolors


class RLTrainer(object):
    def __init__(self, hyperparams, agent, task, load_first=False):
        self.hyperparams = hyperparams
        self.agent = agent
        self.task = task
        self.init_agent()
        if load_first:
            self.agent.load_params()

    def init_agent(self):
        self.agent.set_actions(self.task.get_actions())
        self.agent.set_state_shape(self.task.get_state_shape())
        self.agent.build()

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
            self.task.start_eval_mode()
        print(bcolors.OKGREEN +
              '-' * 40 + '%s Epoch %d' % (epoch_text, epoch_num) + '-' * 40 +
              bcolors.ENDC)
        # total_reward = 0
        epoch_rewards = []
        self.task.start_episode()
        self.agent.start_episode()
        for step_num in range(step_start, step_start + num_frames):
            episode_step_num += 1
            if self.task.episode_is_over():
                epoch_rewards.append(self.task.get_episode_reward())
                # print(bcolors.OKBLUE +
                #       '[[[ Episode %d finished with reward %d ]]]' %
                #       (episode_num, total_reward) +
                #       bcolors.ENDC)
                print(bcolors.OKBLUE +
                      '[[[ Episode %d finished with reward %f ]]]' %
                      (episode_num, self.task.get_episode_reward()) +
                      bcolors.ENDC)
                # total_reward = 0
                episode_step_num = 0
                episode_num += 1
                self.task.start_episode()
                self.agent.start_episode()
            state = self.task.get_state()
            action = self.agent.choose_action(state, step_num)
            self.task.perform_action(action)
            reward = self.task.get_reward()
            next_state = self.task.get_state()
            experience = (state, action, reward, next_state)
            if train:
                self.agent.have_experience(experience)
                self.agent.learn(step_num)
        if not train:
            self.agent.end_eval_mode()
            self.task.end_eval_mode()
        if len(epoch_rewards) > 0:
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
            print('Saving params ...')
            self.agent.save_params()
            step_num += self.hyperparams['frames_per_epoch']
            episode_num += self.run_epoch(epoch_num,
                                          episode_num,
                                          step_num,
                                          self.hyperparams['frames_per_eval'],
                                          train=False)
            step_num += self.hyperparams['frames_per_eval']
