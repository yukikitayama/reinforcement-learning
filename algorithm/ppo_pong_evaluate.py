import gym
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import imageio

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
ENV = 'PongNoFrameskip-v4'
MODEL = '../model/ppo_pong.pth'
VIDEO = '../video/ppo_pong.gif'
# VIDEO = '../video/ppo_pong.mp4'
SCORE = '../object/ppo_pong_score.pkl'
SAVEFIG = '../image/ppo_pong_score.png'
EPISODE = 3
MAX_STEP = 10000
FPS = 30
SEED = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RIGHT = 4
LEFT = 5
MINIBATCH = 1000


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


def visualize_score(score, ma):
    ma_score = pd.Series(score).rolling(ma).mean()
    upper = pd.Series(score).rolling(ma).quantile(0.9, interpolation='linear')
    lower = pd.Series(score).rolling(ma).quantile(0.1, interpolation='linear')
    plt.plot(ma_score)
    plt.fill_between(x=range(len(upper)),
                     y1=upper,
                     y2=lower,
                     alpha=0.3)
    plt.title(f'Proximal Policy Optimization Pong\n'
              f'{ma}-episode moving average of scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig(SAVEFIG)
    plt.show()


def evaluation(env, agent):

    with imageio.get_writer(VIDEO, fps=FPS) as video:

        for i in range(EPISODE):

            # Initialize
            state = env.reset()
            screen = env.render(mode='rgb_array')
            video.append_data(screen)

            # Start episode
            for j in range(MAX_STEP):

                action = agent.act(state)
                state, reward, done, _ = env.step(action)

                screen = env.render(mode='rgb_array')
                video.append_data(screen)

                if done:
                    break

        env.close()

    print('Saved video')


def main():
    # Environment
    env = gym.make(ENV)
    env.seed(SEED)

    # Score result
    score = pickle.load(open(SCORE, 'rb'))

    # Visualize score
    visualize_score(score=score, ma=100)

    # Load trained network
    policy = Policy()
    # policy.load_state_dict(torch.load(MODEL, map_location=device))

    # Make evaluation video
    # evaluation(env=env, agent=agent)


if __name__ == '__main__':
    main()
