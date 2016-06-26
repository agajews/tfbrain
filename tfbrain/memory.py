import blosc

from collections import deque

import random

import numpy as np


class ExperienceReplay(object):
    def __init__(self, hyperparams, dtype=np.uint8):
        self.hyperparams = hyperparams
        self.state_shape = None
        self.dtype = dtype

    def __len__(self):
        return len(self.replay)

    def build(self):
        replay_len = self.hyperparams['experience_replay_len']
        self.replay = deque(maxlen=replay_len)

    def compress(self, array):
        return blosc.compress(array.flatten().tobytes(), typesize=1)

    def decompress(self, array, shape):
        return np.reshape(np.fromstring(
            blosc.decompress(array), dtype=self.dtype),
            self.state_shape)

    def add_experience(self, experience):
        state, action, reward, next_state = experience
        if self.state_shape is None:
            self.state_shape = state.shape
        state = self.compress(state.astype(self.dtype))
        next_state = self.compress(next_state.astype(self.dtype))
        experience = (state, action, reward, next_state)
        self.replay.append(experience)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.replay)), batch_size)
        sample = []
        for index in indices:
            experience = self.replay[index]
            state, action, reward, next_state = experience
            state = self.decompress(state, self.state_shape)
            next_state = self.decompress(next_state, self.state_shape)
            experience = (state, action, reward, next_state)
            sample.append(experience)
        return sample
