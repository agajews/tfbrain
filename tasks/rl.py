from ale_python_interface.ale_python_interface import ALEInterface

# from tfbrain.helpers import bcolors

import numpy as np

# import cv2

from scipy import ndimage


class AtariTask(object):
    def __init__(self, hyperparams, rom_fnm):
        self.hyperparams = hyperparams
        self.state_len = hyperparams['state_len']
        # self.screen_resize = hyperparams['screen_resize']
        self.rom_fnm = rom_fnm
        self.init_ale(display=False)
        self.actions = self.ale.getMinimalActionSet()
        print('Num possible actions: %d' % len(self.actions))
        self.state_shape = (84, 84, self.state_len)

    def init_ale(self, display=False):
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', 123)
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setInt(b'frame_skip', self.hyperparams['frame_skip'])
        if display:
            self.ale.setBool(b'display_screen', True)
        self.ale.loadROM(str.encode(self.rom_fnm))

    # def set_screen_shape(self):
    #     self.start_episode()
    #     # self.perform_action(0)
    #     self.state_shape = self.get_state().shape
    #     print(self.state_shape)
        # self.prev_screen_rgb = self.ale.getScreenRGB()
        # screen = self.ale.getScreenRGB()
        # screen = self.preprocess_screen(screen)
        # self.screen_shape = screen.shape
        # self.state_shape = (self.state_len,) + \
        #     self.screen_shape

    def preprocess_screen(self, screen_rgb):
        # observation = cv2.cvtColor(cv2.resize(
        #     screen_rgb, (84, 110)), cv2.COLOR_BGR2GRAY)
        # observation = observation[26:110, :]
        # ret, observation = cv2.threshold(
        #     observation, 1, 255, cv2.THRESH_BINARY)
        # return np.reshape(observation, (84, 84))
        screen = np.dot(
            screen_rgb, np.array([.299, .587, .114])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.4, 0.525))
        screen.resize((84, 84))
        return screen

    def get_state_shape(self):
        return self.state_shape

    def get_actions(self):
        return list(range(len(self.actions)))

    def start_episode(self):
        self.ale.reset_game()
        self.episode_reward = 0
        self.states = []
        for _ in range(self.state_len):
            self.states.append(self.preprocess_screen(
                self.ale.getScreenRGB()))

    def get_episode_reward(self):
        return self.episode_reward

    def get_state(self):
        curr_state = np.stack(self.states, axis=2)
        return curr_state
        # curr_state = np.zeros(self.state_shape)
        # for i in range(self.state_len):
        #     curr_state[i] = self.states[i]
        # return curr_state

    def perform_action(self, action_ind):
        # print(action_ind)
        action = self.actions[action_ind]
        # self.curr_reward = 0
        # for frame in range(self.hyperparams['frame_skip']):
        #     self.prev_screen = self.ale.getScreenRGB()
        #     self.curr_reward += self.ale.act(action)
        self.curr_reward = self.ale.act(action)
        self.episode_reward += self.curr_reward
        # if not self.curr_reward == 0:
        #     print(bcolors.WARNING +
        #           'Got real reward!' +
        #           bcolors.ENDC)
        if self.curr_reward > 0:
            self.curr_reward = 1
            # print(bcolors.WARNING +
            #       'Got real reward!' +
            #       bcolors.ENDC)
        elif self.curr_reward < 0:
            self.curr_reward = -1
        self.states = self.states[:3]
        screen = self.ale.getScreenRGB()
        # screen = np.maximum(screen, self.prev_screen)
        screen = self.preprocess_screen(screen)
        self.states.insert(0, screen)

    def start_eval_mode(self):
        self.init_ale(display=True)

    def end_eval_mode(self):
        self.init_ale(display=False)

    def get_reward(self):
        return self.curr_reward

    def episode_is_over(self):
        return self.ale.game_over()
