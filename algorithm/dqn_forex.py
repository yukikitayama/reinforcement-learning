import torch
import torch.nn as nn
import numpy as np


class DqnForex(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DqnForex, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.model(x)


class ExperienceReplay:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = {'state': [], 'action': [], 'reward': [],
                       'next_state': [], 'done': []}

    def store(self, experience):
        # Discard the oldest experience if the buffer is full
        if len(self.buffer['state']) >= self.max_size:
            for key in self.buffer.keys():
                self.buffer[key].pop(0)
        # Add new experience at the end
        for key, value in experience.items():
            self.buffer[key].append(value)

    def size(self):
        return len(self.buffer['state'])


def get_action(state, num_actions, model, epsilon):
    # Epsilon
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    # Greedy
    else:
        return np.argmax(model(state))


def update_target(model, target_model):
    model.load_state_dict(target_model.state_dict())
    return model
