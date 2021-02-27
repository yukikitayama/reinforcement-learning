"""
https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_11/chapter-11.ipynb

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
import time
from itertools import count
import gc  # https://stackify.com/python-garbage-collection/
# Parallel
from multiprocessing import Process, Lock
import os
from torchvision.transforms import Resize
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt


# Parameter
N_WORKERS = 2
MAX_EPISODES = 1
LR = 1e-3
POLICY_OPTIMIZER_LR = 0.0005  # Adam optimizer learning rate for policy
VALUE_OPTIMIZER_LR = 0.0007  # RMSprop optimizer learning rate for state value function
ENTROPY_LOSS_WEIGHT = 0.001  # The hyperparameter beta which controls the strength of the entropy regularization term
# from Asynchronous Methods for Deep Reinforcement Learning paprer
POLICY_MAX_GRAD_NORM = 1  # Gradient clip for policy parameter
VALUE_MAX_GRAD_NORM = float('inf')  # Gradient clip for state value function parameter
PATH_SAVE_POLICY = '../model/a3c_breakout_policy.pth'
ENV = 'BreakoutNoFrameskip-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
SEED = 0


class Policy(nn.Module):
    """
    Fully connected discrete action policy. Forward method returns logits with the length number of discrete actions.
    full_pass method is a helper method to produce everything useful during training, including probabilities, actions,A
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

        # unsqueeze(-1) makes add dimension 1 at the end
        # e.g. [] -> [1], [batch size] -> [batch size, 1]
        log_prob = m.log_prob(action).unsqueeze(-1)
        # m.entropy() produces torch.Tensor with the size torch.Size([])
        # With .unsqueeze(-1) -> Size([1])
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


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        """
        arguments are all default arguments of torch.optim.Adam. Params are weights of neural network. Betas are
        coefficients for gradient averages, eps is numerical stability term to denominator, weight_decay is L2 penalty,
        and amsgrad is boolean weather to use AMSGrad.

        Adam is subclass of torch.optim.Optimizer. Optimizer has an instance variable self.param_groups. param_groups is
        a list of dictionary with keys ['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad']. self.state is also
        an instance variable of torch.optim.Optimizer. It is a dictionary of
        """
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        # The below initialization come from source code of torch Adam,
        # https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
        # group is dictionary keys ['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad']
        for group in self.param_groups:
            # p is weight in each layer of nn
            for p in group['params']:
                # state is dict
                state = self.state[p]
                # Initialize state
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        """
        Modification to original Adam step source code,
        https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
        This eventually runs Adam's step, but I don't know why we have this extra lines in step
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Want to update step by workers sharing shared_step?
                # There are two different, step and steps
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        # But does the below not override steps?
        # -> No, because Adam step updates step, not step"s"
        super().step(closure)


class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                                            momentum=momentum, centered=centered)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data).share_memory_()
                if centered:
                    state['grad_avg'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)


class A3CAgent:
    def __init__(self, env, output_dim,
                 shared_policy, shared_state_value_function, shared_policy_optimizer, shared_value_optimizer,
                 device, seed, entropy_loss_weight, policy_max_grad_norm, value_max_grad_norm,
                 max_episodes, path_save_policy,
                 gamma=0.99, policy_instance=None, policy_class=None,
                 state_value_function_class=None):
        self.n_workers = 2

        self.env = env
        self.output_dim = output_dim
        self.policy = policy_instance
        self.policy_class = policy_class
        self.state_value_function_class = state_value_function_class
        self.device = device
        self.seed = seed
        self.entropy_loss_weight = entropy_loss_weight
        self.gamma = gamma

        self.shared_policy = shared_policy
        self.shared_state_value_function = shared_state_value_function
        self.shared_policy_optimizer = shared_policy_optimizer
        self.shared_value_optimizer = shared_value_optimizer
        self.policy_max_grad_norm = policy_max_grad_norm
        self.value_max_grad_norm = value_max_grad_norm

        self.path_save_policy = path_save_policy

        self.stats = {
            'episode': torch.zeros(1, dtype=torch.int).share_memory_(),
            'episode_reward': torch.zeros([max_episodes]).share_memory_(),
            'n_active_workers': torch.zeros(1, dtype=torch.int).share_memory_()
        }

        # Initialize
        self.get_out_signal = None
        self.get_out_lock = None

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

    # Why staticmethod?
    @staticmethod
    def interaction_step(state, env, local_policy, local_state_value_function,
                         logprobs, entropies, rewards, values):
        # TODO: process state
        # TODO: stack state
        action, logprob, entropy = local_policy.full_pass(state)
        next_state, reward, done, info = env.step(action)

        logprobs.append(logprob)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(local_state_value_function(state))

        return next_state, reward, done

    def optimize_model(self, logprobs, entropies, rewards, values, local_policy, local_state_value_function):
        """
        logprobs: A list of torch tensors log of probability of an action taken size([1])
        entropies: A list of torch tensors size([1])
        values: Should be a list of torch tensors size([1]) state value
        """

        # Get size
        T = len(rewards)

        # Calculate return
        # Starts at base**start, and ends with base**stop
        # If T = 10, 0.99 ** 0 = 1, 0.99 ** 1 = 0.99. 0.99 ** 2, ... 0.99 ** 9 = 0.9135
        # endpoint, if true, stop is the last sample. Otherwise it's not included
        discounts = np.logspace(start=0, stop=T, num=T, endpoint=False, base=self.gamma)
        # rewards is [early reward, ..., late reward], because it's appended in list
        # Starts from length T, and ends with length 1
        # Iteratively, discounts list loses the tail part, and rewards list loses the head part, but those two list
        # are the same length in each iteration of for loop
        # To check run the below
        # for t in range(T):
        #     print('discounts', discounts[:T-t])
        #     print('rewards', rewards[t:])
        #     print()
        # Get discounted returns from each time step
        returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])

        # Make everything torch tensor to do gradient descent
        # Why remove the last one?
        # unsqueeze(1) makes [T,] -> [T, 1]
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        # Why remove the last one?
        # unsqueeze(1) makes [T,] -> [T, 1]
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)
        # logprobs is a Python list. Each element has torch tensor size([1]), which from model and unsqueeze(-1)
        # The below makes it torch tensor size([T])
        logprobs = torch.cat(logprobs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        # Actual sampled returns of episodes - estimated state values from model
        value_error = returns - values
        # error multiplied by log policy, but why multiply discounts?
        policy_loss = -(discounts * value_error.detach() * logprobs).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss

        # Gradient ascent for policy
        self.shared_policy_optimizer.zero_grad()
        loss.bachward()
        # clip_grad_norm_ modifies gradients in-place
        torch.nn.utils.clip_grad_norm_(
            parameters=local_policy.parameters(),
            max_norm=self.policy_max_grad_norm
        )
        #
        for param, shared_param in zip(local_policy.parameters(), self.shared_policy.parameters()):
            # ?
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        # Replace local policy with shared policy
        local_policy.load_state_dict(self.shared_policy.state_dict())

        # Gradient descent for state value function
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=local_state_value_function.parameters(),
            max_norm=self.value_max_grad_norm
        )
        for param, shared_param in zip(local_state_value_function.parameters(),
                                       self.shared_state_value_function.parameters()):
            # ?
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_state_value_function.load_state_dict(self.shared_state_value_function.state_dict())

    def work(self, rank):
        """
        rank is an ID for workers
        """
        print(f'Worker PID: {os.getpid()}')
        self.stats['n_active_workers'].add_(1)

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
        # print(f'Hashcode of local state value function: {hash(local_state_value_function)}')

        # Update stats episode in place, and get the previous episode number before updating
        global_episode_idx = self.stats['episode'].add_(1).item() - 1

        while not self.get_out_signal:

            # Initialize for each episode
            state = env.reset()
            # ?
            total_episode_rewards = 0
            # ?
            logprobs = []
            entropies = []
            rewards = []
            values = []

            # count() generate values starting from start and default interval 1
            # This for loop breaks with break by if statement with done
            for step in count(start=1):
                state, reward, done = self.interaction_step(
                    state, env, local_policy, local_state_value_function,
                    logprobs, entropies, rewards, values
                )

                total_episode_rewards += reward

                if done:
                    # TODO: Process state
                    # TODO: Stack state
                    next_value = local_state_value_function(state).detach().item()
                    rewards.append(next_value)

                    # Update policy and state value function at the end of each episode with collected experience
                    self.optimize_model(
                        logprobs, entropies, rewards, values, local_policy, local_state_value_function
                    )

                    # Clear experiences for next episode
                    logprobs = []
                    entropies = []
                    rewards = []
                    values = []

                if done:
                    # Trigger a manual garbage collection process to clean up objects
                    gc.collect()
                    # Break for step in count(start=1)
                    break

            # Get stats of each episode
            # Value of 'episode_reward' is torch tensor size max_episodes, recording total rewards in each element
            self.stats['episode_reward'][global_episode_idx].add_(total_episode_rewards)

            # Save model
            torch.save(local_policy.state_dict(), self.path_save_policy)

            with self.get_out_lock:
                potential_next_global_episode_idx = self.stats['episode'].item()
                if something:
                    self.get_out_signal.add_(1)
                    # Break for while not self.get_out_signal
                    break

                # Else go to another episode
                global_episode_idx = self.stats['episode'].add_(1).item() - 1


        return None

    def train(self):
        print(f'Main process PID: {os.getpid()}')

        self.get_out_lock = Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()

        # Make multiple workers from list comprehension with multiprocessing.Process
        workers = [Process(target=self.work, args=(rank,)) for rank in range(self.n_workers)]
        print(workers)

        # Start subprocesses
        [worker.start() for worker in workers]

        # process IDs
        # [print(f'PID: {worker.pid}') for worker in workers]

        # Wait all the subprocesses to finish
        [worker.join() for worker in workers]

        final_episode = self.stats['episode'].item()

        print('Training complete')
        self.stats['result'] = self.stats['result'].numpy()

        return self.stats['result']

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
    # shared_policy_optimizer = optim.Adam(shared_policy.parameters(), lr=LR)
    shared_state_value_function_optimizer = optim.Adam(
        shared_state_value_function.parameters(),
        lr=LR
    )
    shared_policy_optimizer = SharedAdam(params=shared_policy.parameters(), lr=POLICY_OPTIMIZER_LR)
    shared_value_optimizer = SharedRMSprop(params=shared_state_value_function.parameters(), lr=VALUE_OPTIMIZER_LR)

    # Test optimizer

    # Agent
    # agent = A3CAgent(env=env, policy_instance=policy, device=device)
    # agent = A3CAgent(env=env, policy_class=policy_function, device=device)
    agent = A3CAgent(env=env, output_dim=action_dim,
                     policy_class=Policy,
                     state_value_function_class=StateValueFunction,
                     shared_policy=shared_policy,
                     shared_state_value_function=shared_state_value_function,
                     shared_policy_optimizer=shared_policy_optimizer,
                     shared_value_optimizer=shared_value_optimizer,
                     policy_max_grad_norm=POLICY_MAX_GRAD_NORM,
                     value_max_grad_norm=VALUE_MAX_GRAD_NORM,
                     device=device, seed=SEED, entropy_loss_weight=ENTROPY_LOSS_WEIGHT,
                     max_episodes=MAX_EPISODES,
                     path_save_policy=PATH_SAVE_POLICY)

    # Test agent
    # agent.test()
    agent.test_train()


if __name__ == '__main__':
    main()
