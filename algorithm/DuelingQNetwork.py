import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_output = nn.Linear(64, 1)
        self.advantage_output = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # Action-advantage function
        a = self.advantage_output(x)
        # State-value function
        v = self.value_output(x)
        # Expand v since it is a single value that will be added to a
        v = v.expand_as(a)
        # Action-value function
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q


class DuelingQ(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingQ, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_output = nn.Linear(64, 1)
        self.advantage_output = nn.Linear(64, action_size)

        # GPU management
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        self.device = torch.device(device)
        self.to(self.device)

        if device == 'cuda:0':
            print('Network is initialized with GPU')
        else:
            print('Network is initialized with CPU')

    def forward(self, x):
        x = self._format(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # Action-advantage function
        a = self.advantage_output(x)
        # State-value function
        v = self.value_output(x)
        # Expand v since it is a single value that will be added to a
        v = v.expand_as(a)
        # Action-value function
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q

    def _format(self, state):
        """
        If the state is numpy array, this private method converts it to torch tensor in forward method.
        """
        x = state

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            # Add one dimension to make it 2 dimensional
            x = x.unsqueeze(0)

        return x

    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable

    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)

        return states, actions, new_states, rewards, is_terminals