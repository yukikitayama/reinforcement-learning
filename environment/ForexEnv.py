"""
State
- rate, position (long, short, or no position)

Action
- long, short, do nothing
  - If currently the agent has a long position, only short or do nothing are
    allowed.

Necessary method
- set seed
- reset
- step
- state space
- action space
- make
"""
import numpy as np
import pickle


PATH_01 = '../object/rate_usdjpy.pkl'


class ForexEnv:
    def __init__(self, episode_length):
        self.data = pickle.load(open(PATH_01, 'rb'))
        self.size = len(self.data)
        self.index = 0
        self.episode_length = episode_length
        self.counter = 0

    def set_seed(self, seed_int):
        np.random.seed(seed_int)

    def step(self):
        state = self.data[self.counter + self.index]
        self.counter += 1
        if self.counter < self.episode_length:
            done = False
        else:
            done = True
        return state, done

    def reset(self):
        index = np.random.randint(low=0, high=(self.size - self.episode_length))
        self.index = index
