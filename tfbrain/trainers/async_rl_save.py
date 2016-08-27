from tfbrain.helpers import bcolors
# from memory_profiler import profile
import threading


class GatherThread(threading.Thread):
    def __init__(self, action_chooser, environment, num_frames):
        threading.Thread.__init__(self)
        self.action_chooser = action_chooser
        self.environment = environment
        self.num_frames = num_frames

    def run(self):
        self.thread_rewards = []
        # self.action_chooser.start_episode()
        # self.environment.start_episode()
        for step_num in range(self.num_frames):
            if self.environment.episode_is_over():
                self.thread_rewards.append(
                    self.environment.get_episode_reward())
                print(bcolors.OKBLUE +
                      '[[[ Episode finished with reward %f ]]]' %
                      (self.environment.get_episode_reward()) + bcolors.ENDC)
                self.action_chooser.end_episode()
                self.action_chooser.start_episode()
                self.environment.start_episode()
            state = self.environment.get_state()
            action = self.action_chooser.choose_action(state, step_num)
            self.environment.perform_action(action)
            reward = self.environment.get_reward()
            next_state = self.environment.get_state()
            experience = (state, action, reward, next_state)
            self.action_chooser.have_experience(experience)
        # self.action_chooser.end_episode()


class EvalThread(threading.Thread):
    def __init__(self, action_chooser, environment, num_episodes):
        threading.Thread.__init__(self)
        self.action_chooser = action_chooser
        self.environment = environment
        self.num_episodes = num_episodes

    def run(self):
        self.thread_rewards = []
        # self.action_chooser.start_episode()
        # self.environment.start_episode()
        # self.action_chooser.start_eval_mode()
        self.environment.start_eval_mode()
        step_num = 0
        for episode_num in range(self.num_episodes):
            while not self.environment.episode_is_over():
                state = self.environment.get_state()
                action = self.action_chooser.choose_action(state, step_num)
                self.environment.perform_action(action)
                step_num += 1
            self.thread_rewards.append(
                self.environment.get_episode_reward())
            print(bcolors.OKBLUE +
                  '[[[ Episode finished with reward %f ]]]' %
                  (self.environment.get_episode_reward()) + bcolors.ENDC)
            self.action_chooser.end_episode()
            self.action_chooser.start_episode()
            self.environment.start_episode()
        # self.action_chooser.end_eval_mode()
        self.environment.end_eval_mode()


class AsyncSleepSaveTrainer(object):
    def __init__(self, hyperparams, agent, task, load_first=False):
        self.hyperparams = hyperparams
        self.agent = agent
        self.task = task
        self.init_agent()
        self.init_task()
        if load_first:
            self.agent.load_params()

    def init_agent(self):
        self.agent.set_actions(self.task.get_actions())
        self.agent.set_state_shape(self.task.get_state_shape())
        self.agent.build()
        self.train_action_choosers = []
        for thread in range(self.hyperparams['num_threads']):
            action_chooser = self.agent.build_action_chooser()
            self.train_action_choosers.append(action_chooser)
            action_chooser.start_episode()
        self.eval_action_choosers = []
        for thread in range(self.hyperparams['num_threads']):
            action_chooser = self.agent.build_action_chooser(train=False)
            self.eval_action_choosers.append(action_chooser)
            action_chooser.start_episode()

    def init_task(self):
        self.train_environments = []
        for thread in range(self.hyperparams['num_threads']):
            environment = self.task.create_environment()
            self.train_environments.append(environment)
            environment.start_episode()
        self.eval_environments = []
        for thread in range(self.hyperparams['num_threads']):
            environment = self.task.create_environment()
            self.eval_environments.append(environment)
            environment.start_episode()

    # @profile
    def run_epoch(self,
                  epoch_num,
                  num_frames,
                  train=True):
        if train:
            epoch_text = 'Train'
        else:
            epoch_text = 'Eval'
        print(bcolors.OKGREEN +
              '-' * 40 + '%s Epoch %d' % (epoch_text, epoch_num) + '-' * 40 +
              bcolors.ENDC)
        epoch_rewards = []
        threads = []
        self.agent.update_chooser_weights()
        if train:
            environments = self.train_environments
            action_choosers = self.train_action_choosers
        else:
            environments = self.eval_environments
            action_choosers = self.eval_action_choosers
        for environment, chooser in zip(environments,
                                        action_choosers):
            if train:
                thread = GatherThread(chooser, environment, num_frames)
            else:
                thread = EvalThread(chooser, environment, num_frames)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
            epoch_rewards += thread.thread_rewards
            del thread
        del threads
        if len(epoch_rewards) > 0:
            print(bcolors.OKGREEN +
                  '%s %d had average reward %f' %
                  (epoch_text, epoch_num,
                   sum(epoch_rewards) / len(epoch_rewards)) +
                  bcolors.ENDC)
        if train:
            for chooser in action_choosers:
                self.agent.aggregate_from(chooser)
            self.agent.learn()

    def train_by_epoch(self):
        step_num = 0
        num_frames = int(self.hyperparams['frames_per_epoch'] /
                         self.hyperparams['num_threads'])
        num_eval_episodes = int(self.hyperparams['episodes_per_eval'] /
                                self.hyperparams['num_threads'])
        for epoch_num in range(self.hyperparams['num_epochs']):
            if self.hyperparams['eval']:
                if epoch_num % self.hyperparams['eval_freq'] == 0:
                    self.run_epoch(epoch_num,
                                   num_eval_episodes,
                                   train=False)
            self.run_epoch(epoch_num,
                           num_frames,
                           train=True)
            print('Saving params ...')
            self.agent.save_params()
            step_num += self.hyperparams['frames_per_epoch']
            print('Step num: %d' % step_num)
