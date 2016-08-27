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

    def clear(self):
        self.build()

    def compress(self, array):
        return blosc.compress(array.flatten().tobytes(), typesize=1)

    def decompress(self, array):
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
            state = self.decompress(state)
            next_state = self.decompress(next_state)
            experience = (state, action, reward, next_state)
            sample.append(experience)
        return sample


class RDRLMem(ExperienceReplay):
    def add_experience(self, experience):
        state, action, reward, next_state, value = experience
        if self.state_shape is None:
            self.state_shape = state.shape
            print('Replay state shape')
            print(self.state_shape)
        state = self.compress(state.astype(self.dtype))
        next_state = self.compress(next_state.astype(self.dtype))
        experience = (state, action, reward, next_state, value)
        self.replay.append(experience)

    def sample(self, batch_size, rollout_length, decompress=True):
        indices = random.sample(range(len(self.replay) - rollout_length),
                                batch_size)
        sample = []
        for index in indices:
            experience_seq = []
            for frame_ind in range(index, index + rollout_length):
                experience = self.replay[frame_ind]
                state, action, reward, next_state, value = experience
                if decompress:
                    state = self.decompress(state)
                    next_state = self.decompress(next_state)
                experience = (state, action, reward, next_state, value)
                experience_seq.append(experience)

            sample.append(experience_seq)
        return sample


class SNDQNExperienceReplay(object):
    def __init__(self, hyperparams, dtype=np.uint8):
        self.hyperparams = hyperparams
        self.state_shape = None
        self.dtype = dtype

    def __len__(self):
        return len(self.replay)

    def build(self):
        self.replay = deque()

    def clear(self):
        self.build()

    def compress(self, array):
        return blosc.compress(array.flatten().tobytes(), typesize=1)

    def decompress(self, array):
        return np.reshape(np.fromstring(
            blosc.decompress(array), dtype=self.dtype),
            self.state_shape)

    def add_experience(self, experience):
        state, action, reward, next_state, value = experience
        if self.state_shape is None:
            self.state_shape = state.shape
        state = self.compress(state.astype(self.dtype))
        next_state = self.compress(next_state.astype(self.dtype))
        experience = (state, action, reward, next_state, value)
        self.replay.append(experience)

    def get_all(self):
        sample = []
        for experience in self.replay:
            state, action, reward, next_state, value = experience
            state = self.decompress(state)
            next_state = self.decompress(next_state)
            experience = (state, action, reward, next_state, value)
            sample.append(experience)
        return list(sample)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.replay)), batch_size)
        sample = []
        for index in indices:
            experience = self.replay[index]
            state, action, reward, next_state, value = experience
            state = self.decompress(state)
            next_state = self.decompress(next_state)
            experience = (state, action, reward, next_state, value)
            sample.append(experience)
        return sample
