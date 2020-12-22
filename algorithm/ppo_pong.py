import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import pong_utils

# Parameter
ENV = 'PongDeterministic-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
RIGHT = 4
LEFT = 5


# pong_utils.py
def preprocess_single(image, bkg_color=np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
    return img


def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])


def view_environment(env):
    env.reset()
    _, _, _, _ = env.step(0)
    for _ in range(20):
        frame, _, _, _ = env.step(1)

    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.title('Preprocessed image')
    plt.imshow(preprocess_single(frame), cmap='Greys')
    plt.show()

    print('Shape of preprocessed frame', preprocess_single(frame).shape)


class PolicyTNakae(nn.Module):
    def __init__(self):
        super(PolicyTNakae, self).__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # (80 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1 = 78 / 2 + 1 = 39 + 1 = 40
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, stride=2)
        # (40 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1 = 38 / 2 + 1 = 20
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2)
        # (20 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1 = 18 / 2 + 1 = 10
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
        # (10 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1 = 8 / 2 + 1 = 5
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        self.size = 32 * 5 * 5
        self.fc1 = nn.Linear(self.size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


class Policy(nn.Module):
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
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))


def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.01):
    """
    Clipped surrogate function.
    """
    # List of discount factors to the power of t
    # First element is gamma^0, followed by gamma^1, gamma^2, and so on
    discount = discount ** np.arange(len(rewards))
    # np.newaxis is used to increase dimension
    # rewards shape becomes (len(discount), len(rewards))
    # First row has rewards each multiplied by discount * gamma^0,
    # second row also has reward each multiplied by discount * gamma^1, and third by discount * gamma^2
    rewards = np.asarray(rewards) * discount[:, np.newaxis]

    # Sum of discounted future rewards for each reward and for each time step
    # rewards_future shape has (len(rewards), len(rewards))
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    # Avoid zero division
    std = np.std(rewards_future, axis=1) + 1.0e-10

    # mean[:, np.newaxis] shape has (len(rewards_future), 1) from mean shape (len(rewards_future), )
    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)

    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)

    # Probability ratio
    ratio = new_probs / old_probs

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, min=1 - epsilon, max=1 + epsilon)

    # Clipped surrogate objective
    # (7) of proximal policy optimization algorithms paper
    clipped_surrogate_objective = torch.min(ratio * rewards, clipped_ratio * rewards)

    # ?
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * (torch.log(1.0 - old_probs + 1.e-10)))

    return torch.mean(clipped_surrogate_objective + beta * entropy)


def test_clipped_surrogate():
    rewards = [1, 1, 10]
    discount = 0.995
    discount = discount ** np.arange(len(rewards))
    print('discount\n', discount)
    rewards = np.asarray(rewards) * discount[:, np.newaxis]
    print('rewards\n', rewards, rewards.shape)
    print('rewards[::-1]\n', rewards[::-1])
    print('rewards[::-1].cumsum(axis=0)\n', rewards[::-1].cumsum(axis=0))
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    print('rewards_future\n', rewards_future, rewards_future.shape)

    mean = np.mean(rewards_future, axis=1)
    # Avoid zero division
    std = np.std(rewards_future, axis=1) + 1.0e-10
    print('mean\n', mean, mean.shape)
    print('mean[:, np.newaxis]\n', mean[:, np.newaxis], mean[:, np.newaxis].shape)
    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]
    print('rewards_normalized\n', rewards_normalized, rewards_normalized.shape)


def main():
    env = gym.make(ENV)
    print('Environment', env)
    print('List of available actions:', env.unwrapped.get_action_meanings())

    # View environment
    # view_environment(env)

    # Test clipped_surrogate
    test_clipped_surrogate()



if __name__ == '__main__':
    main()
