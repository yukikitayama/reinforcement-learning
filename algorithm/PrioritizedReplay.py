"""
Github
https://github.com/rlcode/per/blob/master/prioritized_memory.py
https://github.com/rlcode/per/blob/master/SumTree.py
https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/replay_buffer.py

Blog
https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
https://adventuresinmachinelearning.com/prioritised-experience-replay/
https://danieltakeshi.github.io/2019/07/14/per/

Question
- What is the capacity of SumTree
- Where is the sampling_prob showing up
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Parameter

# Deep Q Network
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY_STEP = 20000
EPSILON_DECAY_RATE = 0.999

# experience replay
ALPHA_PER =0.6
BETA_PER = 0.4
EPSILON_PER = 0.01
# EPSILON_PER = 1e-6


class SumTree:
    def __init__(self):



class PrioritizedReplay:
    def __init__(self, capacity,
                 alpha=ALPHA_PER, beta=BETA_PER, epsilon_per=EPSILON_PER):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon_per = epsilon_per

    def sample(self):
        return experience, index, priority

    def update(self, index, priority):
        self.tree.update(index, priority)

    def add(self):
        return None

    def _update_beta(self):

    def __len__(self):
        return self.n_entries


class PERAgent:
    def __init__(self):
        self.online_network
        self.target_network

    def compute_loss(self):
        """
        Compute loss from q target and q estmate
        """
        return None

    def step(self):
        """
        Update weight of neural network
        """

    def train(self, env, seed, gamma, ):

    def update_network(self, tau):

    def save_network(self, model_path, model):


class GreedyAlgorithm:
    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
        return np.argmax(q_values)


class EpsilonGreedyAlgorithm:
    def __init__(self, epsilon_start, epsilon_min,
                 epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_decay_step=None):
        self.epsilon = epsilon_start
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step

    def _epsilon_update(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))

        # Decay epsilon
        self._epsilon_update()

        return action


class DuelingQNetwork(nn.Modelu):
    def __init__(self, output_dim):
        super(DuelingQNetwork, self).__init__()
        # in_channels=2 because 2 frames are stacked
        # (80 + 2 * 0 - 1 * (6 - 1) - 1) / 2 + 1 = (80 - 6) / 2 + 1 = 74 / 2 + 1 = 38
        # From (80 * 80 * 2) to (38 * 38 * 4)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2)
        # (38 + 2 * 0 - 1 * (6 - 1) - 1) / 4 + 1 = (38 - 6) / 4 + 1 = 32 / 4 + 1 = 9
        # From (38 * 38 * 4) to (9 * 9 * 16)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, self.size)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Output logits
        return x


def main():


if __name__ == '__main__':
    main()
