"""
Atari Pong
- When a ball pass our area, agent receive -1.0 as losing a point. Usually, every time step, agent receives 0.0 reward.
"""

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

import random
from collections import deque
import pickle


# Parameter
# EPISODE = 3000
# EPISODE = 100
EPISODE = 1000
K = 5
# MINIBATCH = 1000
MINIBATCH = 500
GAMMA = 0.99
# GAMMA = 0.999
# GAMMA = 0.9995
EPSILON = 0.1
MAXLEN = 100
LR = 1e-3
# LR = 1e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
# ENV = 'PongNoFrameskip-v4'
ENV = 'PongDeterministic-v4'
SEED = 0
# Actions ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
RIGHT = 4  # RIGHTFIRE
LEFT = 5  # LEFTFIRE
NRAND = 5
SCORE = '../object/ppo_pong_score.pkl'
MODEL = '../model/ppo_pong.pth'
TIME = '../object/ppo_pong_time.pkl'
LOSS = '../object/ppo_pong_loss.pkl'


class Policy(nn.Module):
    """
    This outputs the probability of moving right, so P(left) = 1 - P(right)
    """
    def __init__(self):
        super(Policy, self).__init__()
        # in_channels=2 is because of 2 stacked frames
        # (80 + 2 * 0 - 1 * (6 - 1) - 1) / 2 + 1 = (80 - 6) / 2 + 1 = 74 / 2 + 1 = 38
        # 80 * 80 * 2 to 38 * 38 * 4
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2, bias=False)
        # (38 + 2 * 0 - 1 * (6 - 1) - 1) / 4 + 1 = (38 - 6) / 4 + 1 = 32 / 4 + 1 = 9
        # 38 * 38 * 4 to 9 * 9 * 16
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 2)

        self.layers = nn.Sequential(
            nn.Linear(6000 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = x.view(-1, self.size)
    #     x = F.relu(self.fc1(x))
    #     logits = self.fc2(x)
    #     return logits

    def forward(self, x):
        return self.layers(x)


class PPOAgent:
    def __init__(self, env, policy, gamma, epsilon, device, nrand):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.nrand = nrand

    def preprocess_batch(self, images, bkg_color=np.array([144, 72, 17])):
        """
        Returns batch data which has shape (1, 2, 80, 80).
        """
        list_of_images = np.asarray(images)

        if len(list_of_images.shape) < 5:
            list_of_images = np.expand_dims(list_of_images, 1)

        list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color,
                                        axis=-1) / 255.0

        batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
        # Send input data to device to be in the same location as policy
        return torch.from_numpy(batch_input).float().to(self.device)

    def state_to_tensor(self, I):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
        if I is None:
            return torch.zeros(1, 6000)
        I = I[
            35:185]  # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        I = I[::2, ::2, 0]  # downsample by factor of 2.
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0).to(self.device)

    def pre_process(self, x, prev_x):
        # return self.state_to_tensor(x) - self.state_to_tensor(prev_x)
        return torch.cat([self.state_to_tensor(x), self.state_to_tensor(prev_x)], dim=1)

    def play(self):
        return None

    def sample_minibatch(self, m, state_list, action_list, action_prob_list, reward_list):
        idxs = random.sample(range(len(action_list)), m)
        state_batch = torch.cat([state_list[idx] for idx in idxs], dim=0)
        # https://pytorch.org/docs/stable/tensors.html
        action_batch = torch.tensor(data=[action_list[idx] for idx in idxs], dtype=torch.int64, device=self.device)
        action_prob_batch = torch.tensor(data=[action_prob_list[idx] for idx in idxs], dtype=torch.float, device=self.device)
        reward_batch = torch.tensor(data=[reward_list.cpu().numpy()[idx] for idx in idxs], dtype=torch.float, device=self.device)
        return state_batch, action_batch, action_prob_batch, reward_batch

    def clipped_surrogate(self, minibatch):
        state_batch, action_batch, action_prob_batch, reward_batch = minibatch

        # Get logits from policy
        new_logits = self.policy(state_batch)

        # Get probability of each state and each action
        new_probs = F.softmax(new_logits, dim=1)

        # Action index [4, 5] to probability tensor index [0, 1]
        action_batch = self.convert_action_for_prob(action_batch)

        # unsqueeze(input, dim, index) returns a new tensor with a new size one dimension inserted at dim
        new_probs = new_probs.gather(dim=1, index=action_batch.unsqueeze(1)).squeeze()

        # Probability ratio
        ratio = new_probs / action_prob_batch

        # Clipped ratio
        clipped_ratio = torch.clamp(ratio, min=(1 - self.epsilon), max=(1 + self.epsilon))

        # Clipped surrogate objective
        # (7) of proximal policy optimization algorithms paper
        clipped_surrogate_objective = torch.min(ratio * reward_batch,
                                                clipped_ratio * reward_batch)

        loss = torch.mean(clipped_surrogate_objective)

        return loss

    def random_beginning(self):
        self.env.reset()
        for _ in range(self.nrand):
            state1, reward1, _, _ = self.env.step(np.random.choice([RIGHT, LEFT]))
            state2, reward2, _, _ = self.env.step(0)
        return state1, state2

    def convert_action_for_env(self, action):
        return action + 4

    def convert_action_for_prob(self, action):
        return action - 4

    def collect_trajectory(self):
        state_list = []
        action_list = []
        action_prob_list = []
        reward_list = []
        t = 0

        state1, state2 = self.random_beginning()

        while True:

            t += 1

            # batch_input = self.preprocess_batch([state1, state2])
            batch_input = self.pre_process(state1, state2)

            with torch.no_grad():
                # action, action_prob = self.policy(batch_input)
                logits = self.policy(batch_input)
                m = Categorical(logits=logits)
                action = int(m.sample().cpu().numpy()[0])
                action_prob = float(m.probs[0, action].detach().cpu().numpy())

            # print('action', action)
            # print('logits', logits)
            # print('m.probs', m.probs[0, :])

            # Action is 0 or 1, but we want 4 or 5
            action = self.convert_action_for_env(action)

            state1 = state2
            state2, reward, done, _ = self.env.step(action)

            state_list.append(batch_input)
            action_list.append(action)
            action_prob_list.append(action_prob)
            reward_list.append(reward)

            if done:
                print('Last probability', m.probs[0, :], 'time step', t)
                break

        return state_list, action_list, action_prob_list, reward_list, t

    def normalize_reward(self, reward_list):
        R = 0
        discounted_rewards = []

        #    [t0, t1, t2, t3]
        # -> [t3, t2, t1, t0]
        # -> [t3 + (0.99 * 0),
        #     t2 + (0.99 * (t3 + (0.99 * 0))),
        #     t1 + (0.99 * (t2 + (0.99 * (t3 + (0.99 * 0)))))]
        for r in reward_list[::-1]:
            if r != 0:
                R = 0  # scored/lost a point in pong, so reset reward sum?
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        # Rewards normalization because the distribution of rewards shifts as learning happens.
        # Then, learning can be improved if we normalize the rewards
        discounted_rewards = torch.tensor(data=discounted_rewards, dtype=torch.float, device=self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1.0e-10)
        return discounted_rewards

    def test_agent(self):
        state1, state2 = self.random_beginning()
        batch_input = self.preprocess_batch([state1, state2])
        # print(batch_input.size())
        # print('(action, action probability)')
        # print(agent.policy(batch_input))
        trajectory = self.collect_trajectory()
        # print('Length of state_list', len(trajectory[0]),
        #       'Length of action_list', len(trajectory[1]),
        #       'Size of the first element of state_list', trajectory[0][0].size())
        # print('Check rewards')
        # for r in trajectory[3]:
        #     print(r)
        rewards = self.normalize_reward(trajectory[3])
        # print(rewards.size(), rewards.type(), rewards.dtype)
        # for r, nr in zip(trajectory[3], rewards):
        #     print(r, nr)
        minibatch = self.sample_minibatch(
            m=MINIBATCH, state_list=trajectory[0], action_list=trajectory[1],
            action_prob_list=trajectory[2], reward_list=rewards
        )
        # print(f'State minibatch size and type: {minibatch[0].size()}, {minibatch[0].type()}\n',
        #       f'Action minibatch size and type: {minibatch[1].size()}, {minibatch[1].type()}\n',
        #       f'Action prob minibatch size and type: {minibatch[2].size()}, {minibatch[2].type()}\n',
        #       f'Reward minibatch size and type: {minibatch[3].size()}, {minibatch[3].type()}')
        # new_probs = agent.states_to_prob(states=minibatch[0])
        # print('new_probs size', new_probs.size())
        # print(new_probs)
        new_probs = self.clipped_surrogate(minibatch)
        # print(new_probs.size())
        print(new_probs)


def main():

    # Environment
    env = gym.make(ENV)

    # Seed
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Policy
    # Send weight to device to be in the same location as input data
    policy = Policy().to(device)

    # Optimizer
    # optimizer = optim.Adam(policy.parameters(), lr=LR)

    # Agent
    agent = PPOAgent(env=env, policy=policy, gamma=GAMMA, epsilon=EPSILON,
                     device=device, nrand=NRAND)

    # Optimizer
    optimizer = optim.Adam(agent.policy.parameters(), lr=LR)

    # Test
    # agent.test_agent()

    # Initialization for training
    total_rewards = []
    total_rewards_for_average = deque(maxlen=MAXLEN)
    time_steps = []
    losses = []

    # Training
    for iteration in range(EPISODE):

        # Trajectory
        trajectory = agent.collect_trajectory()

        # Total rewards
        total_reward = sum(trajectory[3])
        total_rewards.append(total_reward)
        total_rewards_for_average.append(total_reward)

        # Total time steps
        time_steps.append(trajectory[4])

        # Normalize rewards
        normalized_rewards = agent.normalize_reward(trajectory[3])

        # Optimize surrogate L wrt theta, with K epochs and minibatch size M <= NT
        for _ in range(K):

            mini_batch = agent.sample_minibatch(
                m=MINIBATCH, state_list=trajectory[0], action_list=trajectory[1],
                action_prob_list=trajectory[2], reward_list=normalized_rewards
            )

            # Do not forget - to go gradient ascent
            loss = -agent.clipped_surrogate(mini_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Iteration: {iteration:d}\tLoss: {loss:.3f}')

        losses.append(loss)

        print(f'Iteration: {iteration:d}\t'
              f'Last total reward: {total_rewards[-1]:.1f}\t'
              f'MA reward: {np.mean(total_rewards_for_average):.3f}')

    # Save score
    pickle.dump(total_rewards, open(SCORE, 'wb'))
    print('Saved score')

    # Save model
    torch.save(agent.policy.state_dict(), MODEL)
    print('Saved model')

    # Save time steps
    pickle.dump(time_steps, open(TIME, 'wb'))
    print('Saved time steps')

    # Saved loss
    pickle.dump(losses, open(LOSS, 'wb'))
    print('Saved loss')


if __name__ == '__main__':
    main()
