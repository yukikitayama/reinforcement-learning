import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import matplotlib.pyplot as plt
import gym
import pickle


print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
ENV = 'CartPole-v0'
LR = 0.001
GAMMA = 0.99
MAX_EPISODE = 3000
MA = 100
# RENDER = True
RENDER = False
ACTOR = '../model/a2c_actor_cartpole.pt'
CRITIC = '../model/a2c_critic_cartpole.pt'
REWARD = '../object/a2c_cartpole_reward.pkl'
SAVEFIG_01 = '../image/a2c_cartpole_reward.png'


def process_state(state):
    x = torch.from_numpy(state).float()
    x = x.unsqueeze(dim=0)
    return x


# Policy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


# State value function
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def run(env, actor, critic, optimizer_actor, optimizer_critic, gamma, max_episode, render):

    total_rewards = []

    for i in range(max_episode):

        # Initialize
        total_reward = 0
        state = env.reset()

        while True:

            if render:
                env.render()

            # Choose action
            processed_state = process_state(state)
            probs = actor(processed_state)
            m = Categorical(probs)
            action = m.sample()

            # Get next state, reward, and done
            next_state, reward, done, _ = env.step(action.detach().data.numpy()[0])

            # Collect reward
            total_reward += reward

            # Break the while loop
            if done:
                break

            # Calculate advantage
            state_value = critic(processed_state)
            processed_next_state = process_state(next_state)
            target_value = reward + (1 - done) * gamma * critic(processed_next_state)
            advantage = target_value - state_value

            # Update critic
            critic_loss = advantage.pow(2).mean()
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Update actor
            actor_loss = -m.log_prob(action) * advantage.detach()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Go to the next state
            state = next_state

        # Collect episode reward
        total_rewards.append(total_reward)

        # Monitor
        print(f'Episode: {i}, total reward: {total_reward}')

    env.close()

    return total_rewards


def visualize_result(rewards, ma):
    ma_rewards = pd.Series(rewards).rolling(ma).mean()
    plt.plot(ma_rewards)
    plt.title(f'Advantage actor critic in {ENV}')
    plt.xlabel('Episode')
    plt.ylabel('Moving average of rewards')
    plt.grid()
    plt.tight_layout()
    plt.savefig(SAVEFIG_01)
    plt.close()


def main():

    # Environment
    env = gym.make(ENV)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Actor critic
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    # Optimizer
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR)

    # Training
    total_rewards = run(env, actor, critic, optimizer_actor, optimizer_critic, GAMMA, MAX_EPISODE, RENDER)

    # Visualize result
    visualize_result(total_rewards, ma=MA)

    # Save results
    torch.save(actor.state_dict(), ACTOR)
    torch.save(critic.state_dict(), CRITIC)
    pickle.dump(total_rewards, open(REWARD, 'wb'))


if __name__ == '__main__':
    main()
