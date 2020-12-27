"""
Atari Breakout state preprocessing from (210, 160, 3) to ()
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from multiprocessing import Process


# Parameter
ENV = 'BreakoutNoFrameskip-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Policy(nn.Module):
    """
    Fully connected discrete action policy. Forward method returns logits with the length number of discrete actions.
    full_pass method is a helper method to produce everything useful during training, including probabilities, actions,
    entropy
    """
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.input_layer = nn.Linear(input_dim, 32)
        self.hidden_layer = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, output_dim)

    def forward(self, state):
        x = self._format(state)
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x

    def full_pass(self, state):
        logits = self.forward(state)
        m = Categorical(logits=logits)
        action = m.sample()
        log_prob = m.log_prob(action)

        return log_prob

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
            # unsqueeze adds one dimension
            x = x.unsqueeze(0)
        return x

    def test(self):
        env = gym.make(ENV)
        state = env.reset()
        tmp = self.full_pass(state)
        print(tmp)


class A3CAgent:
    def __init__(self):
        self.n_workers = 2

    def work(self, rank):
        """
        rank is an ID for workers
        """
        # Local environment
        env = gym.make(ENV)
        nS = env.observation_space.shape[0]
        nA = env.action_space.n

        # Local seed
        local_seed = self.seed + rank
        env.seed(local_seed)
        torch.manual_seed(local_seed)
        np.random.seed(local_seed)
        random.seed(local_seed)

        # Policy
        # local_policy_model = self.

        # Value function

    def train(self):
        # Make multiple workers from list comprehension with multiprocessing.Process
        workers = [Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]

        return workers

    def test(self):
        return None


def main():

    # Environment
    env = gym.make(ENV)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    print(f'Environment: {env}')
    print(f'State dimension: {state_dim}')
    print(f'Action dimension: {action_dim}')
    print()

    # Policy
    policy = Policy(input_dim=4, output_dim=2).to(device)

    # Test policy
    # policy.test()

    # Agent
    agent = A3CAgent()

    # Test agent
    # print(agent.train())


if __name__ == '__main__':
    main()
