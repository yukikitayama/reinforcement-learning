import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
# User defined
from algorithm.duel_ddqn_per_space_invaders import Agent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

SEED = 0
MODEL_01 = '../model/duel_ddqn_per_space_invaders_online_model_v3.pth'
MODEL_02 = '../model/duel_ddqn_per_space_invaders_target_model_v2.pth'
# MODEL_02 = '../model/duel_ddqn_per_space_invaders_target_model_v3.pth'
SCORE = '../object/duel_ddqn_per_space_invaders_score_v2.pkl'
SAVEFIG_01 = '../image/d3qn_space_invaders_score_01.png'
SAVEFIG_02 = '../image/d3qn_space_invaders_score_02.png'
VIDEO = '../video/d3qn_space_invaders.gif'
ENV = 'SpaceInvaders-v0'
METHOD = 'D3QN PER'
NUM_FRAMES = 4
RESIZE = 84
FPS = 30
EPSION = 0.1


def visualize_training(score, ma, q, savefig):
    ma_score = pd.Series(score).rolling(ma).mean()
    ma_lower = pd.Series(score).rolling(ma).quantile(quantile=q, interpolation='linear')
    ma_upper = pd.Series(score).rolling(ma).quantile(quantile=1 - q, interpolation='linear')
    plt.plot(ma_score, label=f'{ma} moving average of score')
    plt.fill_between(x=range(len(ma_score)), y1=ma_lower, y2=ma_upper, alpha=0.2,
                     label=f'{q} to {1-q} qunatile score')
    plt.title(f'{METHOD} {ENV} training result')
    plt.xlabel('Episode')
    plt.ylabel('Total reward per episode')
    plt.grid()
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()


def choose_action(model, state):
    if random.random() < EPSION:
        action = random.randint(0, 5)
        print('random', action)
        return action
    else:
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = model(state).cpu().detach().data.numpy().squeeze()
        action = np.argmax(q_values)
        print(q_values, action)
        return action


def evaluation(agent, env, model, episode=1):

    with imageio.get_writer(VIDEO, fps=FPS) as video:

        for i in range(episode):

            state = env.reset()
            state = agent.process_state(state)
            state_stack = torch.tensor(
                data=np.repeat(state, NUM_FRAMES).reshape((NUM_FRAMES, RESIZE, RESIZE)),
                dtype=torch.float,
                device=device
            )

            while True:

                env.render()
                video.append_data(env.render(mode='rgb_array'))

                # print(f'state_stack: {torch.sum(state_stack):,.0f}')

                action = choose_action(model=model,
                                       state=state_stack)

                next_state, _, done, _ = env.step(action)

                if done:
                    break

                next_state = agent.process_state(next_state)
                state_stack = agent.process_state_stack(state_stack=state_stack,
                                                        state=next_state)

        env.close()


class DuelingQNetwork(nn.Module):
    def __init__(self, output_dim):
        super(DuelingQNetwork, self).__init__()
        # (84 + 2 * 0 - 1 * (8 - 1) - 1) / 4 + 1 = (84 - 8) / 2 + 1 = 76 / 2 + 1 = 38 + 1 = 39
        # From (84 * 84 * 3) to (39 * 39 * 32)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        # (39 + 2 * 0 - 1 * (4 - 1) - 1) / 2 + 1 = (39 - 4) / 2 + 1 = 35 / 2 + 1 = 17.5 + 1 = 18
        # From (39 * 39 * 32) to (18 * 18 * 64)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        # (18 + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1 = (18 - 3) / 1 + 1 = 15 + 1 = 16
        # From (18 * 18 * 64) to (16 * 16 * 64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        # self.size = 16 * 16 * 64
        # self.fc1 = nn.Linear(self.size, 512)
        self.fc1 = nn.Linear(3136, 512)
        self.state_value = nn.Linear(512, 1)
        self.action_advantage = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # x = x.view(-1, self.size)  # Flatten
        x = x.reshape(-1, 3136)
        x = self.fc1(x)
        x = F.relu(x)
        # Action-advantage
        a = self.action_advantage(x)
        # State-value
        v = self.state_value(x).expand_as(a)
        # Action-value
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q


def main():

    # Score
    score = pickle.load(open(SCORE, 'rb'))

    # Visualize training
    # visualize_training(score=score, ma=100, q=0.1, savefig=SAVEFIG_01)
    # visualize_training(score=score, ma=1000, q=0.1, savefig=SAVEFIG_02)

    # Environment
    env = gym.make(ENV)

    # Load trained model
    online_model = DuelingQNetwork(output_dim=env.action_space.n)
    online_model.load_state_dict(torch.load(MODEL_01, map_location='cuda:0'))
    online_model.to(device)
    online_model.eval()

    target_model = DuelingQNetwork(output_dim=env.action_space.n)
    target_model.load_state_dict(torch.load(MODEL_02, map_location='cuda:0'))
    target_model.to(device)
    target_model.eval()

    # Agent
    agent = Agent(num_actions=env.action_space.n,
                  online_model=None, target_model=None, optimizer=None)

    # Evaluate
    env.seed(SEED)
    evaluation(agent=agent, env=env, model=online_model, episode=10)


if __name__ == '__main__':
    main()