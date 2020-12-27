"""
Atari Breakout state preprocessing from (210, 160, 3) to ()
"""
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
# Parallel
from multiprocessing import Process
import os
from torchvision.transforms import Resize
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt


# Parameter
LR = 1e-3
ENV = 'BreakoutNoFrameskip-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 0


class Policy(nn.Module):
    """
    Fully connected discrete action policy. Forward method returns logits with the length number of discrete actions.
    full_pass method is a helper method to produce everything useful during training, including probabilities, actions,
    entropy
    """
    def __init__(self, output_dim):
        super(Policy, self).__init__()
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

    def full_pass(self, state):
        logits = self.forward(state)
        m = Categorical(logits=logits)
        action = m.sample()

        # unsqueeze(-1) makes (minibatch size, 1)
        log_prob = m.log_prob(action).unsqueeze(-1)
        entropy = m.entropy().unsqueeze(-1)

        return action, log_prob, entropy

    def select_action(self, state):
        logits = self.forward(state)
        m = Categorical(logits=logits)
        action = m.sample()
        return action

    def select_greedy_action(self, state):
        logits = self.forward(state)
        actions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        return actions

    def test(self):
        return None


class StateValueFunction(nn.Module):
    def __init__(self):
        super(StateValueFunction, self).__init__()
        # in_channels=2 because 2 frames are stacked
        # (80 + 2 * 0 - 1 * (6 - 1) - 1) / 2 + 1 = (80 - 6) / 2 + 1 = 74 / 2 + 1 = 38
        # From (80 * 80 * 2) to (38 * 38 * 4)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2)
        # (38 + 2 * 0 - 1 * (6 - 1) - 1) / 4 + 1 = (38 - 6) / 4 + 1 = 32 / 4 + 1 = 9
        # From (38 * 38 * 4) to (9 * 9 * 16)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)  # Outputs state-value

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, self.size)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Outputs state-value
        return x


class A3CAgent:
    def __init__(self, env, output_dim,
                 shared_policy, shared_state_value_function,
                 device, seed, policy_instance=None, policy_class=None,
                 state_value_function_class=None):
        self.n_workers = 2

        self.env = env
        self.output_dim = output_dim
        self.policy = policy_instance
        self.policy_class = policy_class
        self.state_value_function_class = state_value_function_class
        self.device = device
        self.seed = seed

        self.shared_policy = shared_policy
        self.shared_state_value_function = shared_state_value_function

    def process_state(self, state):
        x = torch.tensor(data=state, dtype=torch.float, device=device)
        # Change color channel position from (210, 160, 3) to (1, 210, 160)
        x = x.permute(2, 0, 1)
        # From color to gray
        x = rgb_to_grayscale(x)
        # Resize from (1, 210, 160) to (1, 80, 80)
        x = Resize([80, 80])(x)
        # Reduce size 1 dimension
        x = x.squeeze(0)

        return x

    def stack_state(self, state, next_state):
        x = torch.stack([next_state, state])
        # Add batch size dimension
        x = x.unsqueeze(0)

        return x

    def work(self, rank):
        """
        rank is an ID for workers
        """
        print(f'Worker PID: {os.getpid()}')

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
        local_policy = self.policy_class(output_dim=self.output_dim)
        local_policy.load_state_dict(self.shared_policy.state_dict())
        # print(f'Hashcode of local policy: {hash(local_policy)}')

        # Value function
        local_state_value_function = self.state_value_function_class()
        local_state_value_function.load_state_dict(self.shared_state_value_function.state_dict())
        print(f'Hashcode of local state value function: {hash(local_state_value_function)}')

        return None

    def train(self):
        print(f'Main process PID: {os.getpid()}')

        # Make multiple workers from list comprehension with multiprocessing.Process
        workers = [Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]
        print(workers)

        # Start subprocesses
        [worker.start() for worker in workers]

        # process IDs
        # [print(f'PID: {worker.pid}') for worker in workers]

        # Wait all the subprocesses to finish
        [worker.join() for worker in workers]

        return None

    def test(self):
        # process_state
        # state = self.env.reset()
        # x = self.process_state(state)
        # print('Processed size', x.size())
        # plt.imshow(x.cpu().numpy(), cmap='gray')
        # plt.show()

        # stack_state
        state1 = self.env.reset()
        state2, _, _, _ = self.env.step(0)
        state1 = self.process_state(state1)
        state2 = self.process_state(state2)
        # print('state1.size()', state1.size())
        # print('state2.size()', state2.size())
        x = self.stack_state(state1, state2)
        # print('stacked states size', x.size())

        # Policy forward
        self.policy.eval()

        with torch.no_grad():
            logits = self.policy(x)
        # print('logits', logits)
        # print('probability', F.softmax(logits, dim=1))

        # print('x.size()', x.size())
        x2 = torch.cat([x, x], dim=0)
        # print('x2.size()', x2.size())
        with torch.no_grad():
            logits = self.policy(x2)
        # print('logits', logits)
        # print('logits.size()', logits.size())

        # Policy full_pass
        with torch.no_grad():
            action, log_prob, entropy = self.policy.full_pass(x2)
        # print('Action\n', action)
        # print('Log probability\n', log_prob)
        # print('Entropy\n', entropy)

        # Policy select_action
        with torch.no_grad():
            actions = self.policy.select_action(x2)
        # print('Actions\n', actions)

        # Policy select_greedy_action
        with torch.no_grad():
            actions = self.policy.select_greedy_action(x2)
        # print('Actions\n', actions)

        self.policy.train()

        return None

    def test_train(self):
        self.train()

        return None


def main():

    # Environment
    env = gym.make(ENV)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    # print(f'Environment: {env}')
    # print(f'State dimension: {state_dim}')
    # print(f'Action dimension: {action_dim}')
    # print(f'Available actions: {env.unwrapped.get_action_meanings()}')
    # print()

    # Seed
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Policy
    # policy = Policy(output_dim=action_dim).to(device)
    # policy_function = lambda x: Policy(x).to(device)
    shared_policy = Policy(output_dim=action_dim).to(device).share_memory()

    # State value function
    shared_state_value_function = StateValueFunction().to(device).share_memory()

    # Optimizer
    shared_policy_optimizer = optim.Adam(shared_policy.parameters(), lr=LR)
    shared_state_value_function_optimizer = optim.Adam(
        shared_state_value_function.parameters(),
        lr=LR
    )

    # Agent
    # agent = A3CAgent(env=env, policy_instance=policy, device=device)
    # agent = A3CAgent(env=env, policy_class=policy_function, device=device)
    agent = A3CAgent(env=env, output_dim=action_dim,
                     policy_class=Policy,
                     state_value_function_class=StateValueFunction,
                     shared_policy=shared_policy,
                     shared_state_value_function=shared_state_value_function,
                     device=device, seed=SEED)

    # Test agent
    # agent.test()
    agent.test_train()


if __name__ == '__main__':
    main()
