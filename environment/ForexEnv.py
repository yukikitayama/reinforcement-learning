"""
State
- rate
- position, 0: no position, 1: long, 2: short

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
    # Class variable
    data = pickle.load(open(PATH_01, 'rb'))
    size = len(data)
    num_states = 2
    num_actions = 3

    def __init__(self, episode_length):
        # Instance variable
        self.index = 0
        self.episode_length = episode_length
        self.counter = 0
        self.position = 0
        self.rate = 0

    # String representation for debugging
    def __str__(self):
        return ('ForexEnv('
                + 'counter=' + str(self.counter)
                + ', position=' + str(self.position)
                + ', rate=' + str(round(self.rate, 2))
                + ', index=' + str(self.index)
                + ')')

    def set_seed(self, seed_int):
        np.random.seed(seed_int)

    def step(self, action):
        # Current rate
        curr_rate = self.data[self.index + self.counter]

        # Next rate
        self.counter += 1
        next_rate = self.data[self.index + self.counter]

        # Done
        if self.counter < self.episode_length:
            done = False
        else:
            done = True

        # Reward, position, position rate
        if action == self.position:
            reward = 0
        # Make position
        elif self.position == 0 and (action == 1 or action == 2):
            reward = 0
            self.position = action
            self.rate = curr_rate
        # Close long position
        elif self.position == 1 and action == 0:
            reward = curr_rate - self.rate
            self.position = 0
            self.rate = 0
        # Close short position
        elif self.position == 2 and action == 0:
            reward = self.rate - curr_rate
            self.position = 0
            self.rate = 0
        # Not allowed action
        elif (self.position == 1 and action == 2) or (self.position == 2 and action == 1):
            reward = 0
        else:
            reward = 0

        return [next_rate, self.position], reward, done

    def reset(self):
        # Randomly initialize beginning index
        index = np.random.randint(low=0, high=(self.size - self.episode_length))
        self.index = index
        self.counter = 0
        self.position = 0
        self.rate = 0
        return self.data[self.index]
