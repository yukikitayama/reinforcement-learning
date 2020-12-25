import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

from collections import deque
import pickle

# Parameter
# N_EPISODES = 1000
N_EPISODES = 10000
PRINT_EVERY = 100
MAX_T = 1000
# GAMMA = 1.0
GAMMA = 0.99
LR = 1e-2
# LR = 1e-4
H_SIZE = 16
# H_SIZE = 256
ENV = 'CartPole-v0'
# ENV = 'MountainCar-v0'
POLICY = '../model/REINFORCE_' + ENV + '_policy.pth'
SCORE = '../object/REINFORCE_' + ENV + '_score.pkl'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
SEED = 0
MAXLEN_DEQUE = 100
THRESHOLD = 195.0  # CartPole-v0
# THRESHOLD = -110.0  # MountainCar-v0


class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Return probability of each action
        return F.softmax(x, dim=1)

    def act(self, state):
        # unsqueeze(0) changes (4,) into (1, 4)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        # https://pytorch.org/docs/stable/distributions.html
        m = Categorical(probs)
        action = m.sample()
        # action.item() is used in env.step(), and m.log_prob(action) is used to calculate loss
        return action.item(), m.log_prob(action)


def reinforce(env, policy, optimizer,
              n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    # Initialization for training
    # scores_deque is used for monitoring and solving threshold
    scores_deque = deque(maxlen=MAXLEN_DEQUE)
    scores = []

    for i_episode in range(1, n_episodes + 1):

        # Initialization for each episode
        saved_log_probs = []
        rewards = []
        state = env.reset()

        for t in range(max_t):

            # action is sample action from policy probability output
            # log_prob is log of the probability of the action, used for loss calculation
            action, log_prob = policy.act(state)

            saved_log_probs.append(log_prob)

            state, reward, done, _ = env.step(action)

            rewards.append(reward)

            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # First element is the beginning of the episode, and the last element is the end of episode
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        discounted_rewards = [gamma_t * reward_t for gamma_t, reward_t in zip(discounts, rewards)]
        # R is scalar
        R = sum(discounted_rewards)

        policy_loss = []
        for log_prob in saved_log_probs:
            # policy_loss is a list of tensors
            policy_loss.append(-log_prob * R)

        # torch.cat concatenates each tensor in tuple or list
        # policy_loss was originally list of tensors but now it is a scalar
        policy_loss = torch.cat(policy_loss).sum()

        # Gradient ascent because policy_loss is sum of negative log of probability
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Monitoring
        if i_episode % print_every == 0:
            print(f'Episode {i_episode}\tAverage score: {np.mean(scores_deque):.2f}')

        # Solving criteria
        if np.mean(scores_deque) >= THRESHOLD:
            print(f'Solved environment in {(i_episode - MAXLEN_DEQUE):d} episode'
                  f'\tAverage score: {np.mean(scores_deque):.2f}')
            torch.save(policy.state_dict(), POLICY)
            print('Saved policy')
            break

    return scores


def main():

    # Environment
    env = gym.make(ENV)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    print('Environment', env)
    print('Observation space', env.observation_space)
    print('Action space', env.action_space)
    print()

    # Seed
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Policy
    policy = Policy(s_size=s_size, h_size=H_SIZE, a_size=a_size).to(device)

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # Training
    scores = reinforce(env=env, policy=policy, optimizer=optimizer,
                       n_episodes=N_EPISODES, max_t=MAX_T,
                       gamma=GAMMA, print_every=PRINT_EVERY)

    # Save score
    pickle.dump(scores, open(SCORE, 'wb'))
    print('Saved score')


if __name__ == '__main__':
    main()
