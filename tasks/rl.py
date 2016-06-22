from ale_python_interface import ALEInterface

import matplotlib.pyplot as plt

import numpy as np

from scipy.misc import imresize


class AtariTask(object):
    def __init__(self, hyperparams, rom_fnm):
        self.hyperparams = hyperparams
        self.state_len = hyperparams['state_len']
        self.screen_resize = hyperparams['screen_resize']
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', 123)
        self.ale.loadROM(str.encode(rom_fnm))
        self.display_screen_every_time = False
        screen = self.ale.getScreenGrayscale()
        print(screen.shape)
        screen = self.preprocess_screen(screen)
        print(screen.shape)
        self.screen_shape = screen.shape
        self.state_shape = (self.state_len,) + \
            self.screen_shape
        self.actions = self.ale.getMinimalActionSet()
        print('Num possible actions: %d' % len(self.actions))

    def preprocess_screen(self, screen):
        screen = np.reshape(screen, screen.shape[:2])
        screen = imresize(screen, self.screen_resize)
        screen = np.reshape(screen, screen.shape + (1,))
        return screen

    def get_state_shape(self):
        return self.state_shape

    def get_actions(self):
        return list(range(len(self.actions)))

    def start_episode(self):
        self.ale.reset_game()
        self.states = []
        for _ in range(self.state_len):
            self.states.append(np.zeros(self.screen_shape))

    def display_screen(self):
        screen = self.ale.getScreenRGB()
        print(screen.shape)
        print(screen)
        plt.imshow(screen)
        plt.show(block=True)
        # self.display_screen = False

    def get_state(self):
        if self.display_screen_every_time:
            self.display_screen()
        self.states.pop(0)
        screen = self.ale.getScreenGrayscale()
        screen = self.preprocess_screen(screen)
        self.states.append(screen)
        curr_state = np.zeros(self.state_shape)
        for i in range(self.state_len):
            curr_state[i] = self.states[i]
        return curr_state

    def perform_action(self, action_ind):
        action = self.actions[action_ind]
        self.curr_reward = 0
        for frame in range(self.hyperparams['frame_skip']):
            self.curr_reward += self.ale.act(action)
        if self.curr_reward > 0:
            self.curr_reward = 1
        elif self.curr_reward < 0:
            self.curr_reward = -1

    def get_reward(self):
        return self.curr_reward

    def episode_is_over(self):
        return self.ale.game_over()
