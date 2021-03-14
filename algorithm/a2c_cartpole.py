import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

ENV = 'CartPole-v0'
HIDDEN_SIZE = 256
# LEARNING_RATE = 3e-4  # 0.0003
LEARNING_RATE = 0.0001
MAX_EPISODES = 3000


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # State value function (Critic)
        self.critic_1 = nn.Linear(state_dim, hidden_size)
        self.critic_2 = nn.Linear(hidden_size, 1)
        # Policy (Actor)
        self.actor_1 = nn.Linear(state_dim, hidden_size)
        self.actor_2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        # Convert numpy to torch tensor size (batch size, state dimension)
        x = torch.from_numpy(state).to(torch.float32).unsqueeze(0)
        # State value function
        value = self.critic_1(x)
        value = F.relu(value)
        value = self.critic_2(value)
        # Policy
        probs = self.actor_1(x)
        probs = F.relu(probs)
        probs = self.actor_2(probs)
        probs = F.softmax(probs, dim=1)
        return value, probs


def main():

    # Environment
    env = gym.make(ENV)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Network
    network = ActorCritic(state_dim, action_dim, HIDDEN_SIZE)

    # Optimizer
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

    buffer_log_probs = []
    buffer_values = []
    buffer_rewards = []
    state = env.reset()

    value, policy_dist = network.forward(state)
    value = value.detach().numpy()[0, 0]
    probs = policy_dist.detach().numpy()
    print(f'value: {value}, probs: {probs}')

    probs = probs.squeeze(0)

    # action = np.random.choice(action_dim, p=np.squeeze(probs))
    action = np.random.choice(action_dim, p=probs)
    print(f'action: {action}')

    # Log of probability given an action
    log_prob = torch.log(policy_dist.squeeze(dim=0)[action])
    print(f'log_prob: {log_prob}')

    # Entropy
    entropy = -np.sum(probs * np.log(probs))
    print(f'entropy: {entropy}')

    # Step
    next_state, reward, done, _ = env.step(action)



if __name__ == '__main__':
    main()
