"""
State
- Rate
- Position
  - 0: Short
  - 1: Flat
  - 2: Long

Action
- 0: Do nothing
- 1: Open short
- 2: Open long
- 3: Close

Author: Yuki Kitayama
Date: 2020-10-25
"""
import numpy as np
import pickle


PATH_01 = '../object/rate_usdjpy.pkl'


class ForexEnv:
    # Class variable
    data = pickle.load(open(PATH_01, 'rb'))
    size = len(data)
    num_states = 2
    num_actions = 4

    def __init__(self, length):
        # Instance variable
        self.index = 0
        self.length = length
        self.time = 0
        self.position = 1
        self.rate = 0

    # String representation for debugging
    def __str__(self):
        return ('ForexEnv('
                + 'time=' + str(self.time)
                + ', position=' + str(self.position)
                + ', rate=' + str(round(self.rate, 2))
                + ', index=' + str(self.index)
                + ')')

    def set_seed(self, seed_int):
        np.random.seed(seed_int)


    def step(self, action):
        # Current rate
        curr_rate = self.data[self.index + self.time]

        # Next rate
        self.time += 1
        next_rate = self.data[self.index + self.time]

        # Done
        if self.time < self.length:
            done = False
        else:
            done = True

        # Reward, position, rate
        # Do nothing
        if action == 0:
            reward = 0
            # No update position and rate
        # Open short position
        elif self.position == 1 and action == 1:
            reward = 0
            self.position = 0
            self.rate = curr_rate
        # Open long position
        elif self.position == 1 and action == 2:
            reward = 0
            self.position = 2
            self.rate = curr_rate
        # Close short position
        elif self.position == 0 and action == 3:
            reward = self.rate - curr_rate
            self.position = 1
            self.rate = 0
        # Close long position
        elif self.position == 2 and action == 3:
            reward = curr_rate - self.rate
            self.position = 1
            self.rate = 0
        else:
            reward = 0

        return [next_rate, self.position], reward, done

    def reset(self):
        # Randomly initialize beginning index
        index = np.random.choice(self.size - self.length)
        self.index = index
        self.time = 0
        self.position = 1
        self.rate = 0
        return self.data[self.index], self.position
