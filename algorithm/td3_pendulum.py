import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
ENV = 'Pendulum-v0'
SEED = 0
MAX_SIZE = 10000
LR = 0.001
NOISE = 0.1
BATCH_SIZE = 100
DISCOUNT = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
ACTOR = f'../model/td3_{ENV}.pth'
CRITIC = f'../model/td3_{ENV}.pth'


class Actor(nn.Module):
    """
    Returns the continuous action
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    """
    x is next state
    u is next action
    returns 2 Q-values from the first critic and the second critic at the same time, while get_Q only returns the
    Q-value from the first critic.
    The critic takes an input states and actions concatenated together and outputs a Q-value. DQN outputs Q-values for
    each action, but because TD3 critic inputs the state and an action, it outputs one Q-value for the action.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # First critic
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        # Second critic
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        # First critic
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        # Second critic
        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def get_q1(self, x, u):
        xu = torch.cat([x, u], dim=1)
        # Use the first critic
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class ReplayBuffer:
    """
    ptr move the position to store the data to the first position of the list when the list is full
    """
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(low=0, high=len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), \
               np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)


class TD3:
    """
    Agent
    """
    def __init__(self, state_dim, action_dim, max_action, env):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.max_action = max_action
        self.env = env

    def select_action(self, state, noise=NOISE):
        # reshape(1, -1) adds a batch dimension
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # flatten() converts (1, action dimension) to (action dimension,)
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise != 0:
            action = action + np.random.normal(loc=0, scale=noise, size=self.env.action_space.shape[0])

        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, replay_buffer, iterations, batch_size=BATCH_SIZE, discount=DISCOUNT, tau=TAU,
              policy_noise=POLICY_NOISE, noise_clip=NOISE_CLIP, policy_freq=POLICY_FREQ):
        """
        Arguments:
            iterations (int): how many times to run training
        """
        for it in range(iterations):

            x, y, u, r, d = replay_buffer.sample(batch_size)


    def save(self):
        torch.save(self.actor.state_dict(), ACTOR)
        torch.save(self.critic.state_dict(), CRITIC)


def main():

    # Environment
    env = gym.make(ENV)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f'Environment: {env}, state: {state_dim}, action: {action_dim}, max action: {max_action}')

    # Seed
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Policy
    policy = TD3(state_dim, action_dim, max_action, env)


if __name__ == '__main__':
    main()
