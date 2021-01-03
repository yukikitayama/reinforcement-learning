"""
Github
https://github.com/rlcode/per/blob/master/prioritized_memory.py
https://github.com/rlcode/per/blob/master/SumTree.py
https://github.com/Guillaume-Cr/lunar_lander_per/blob/master/replay_buffer.py

Blog
https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
https://adventuresinmachinelearning.com/prioritised-experience-replay/
https://danieltakeshi.github.io/2019/07/14/per/

Question
- What is the capacity of SumTree?
- Where is the sampling_prob showing up?
- What is getattr()?
  - Get the value of class or instance variables. If not, you can make a new instance variable and set the value from
    outside of the object by setattr()
- What is the difference of index and data_index?
- Why do we sample from uniform distribution in sample method of prioritized replay class?

Algorithm
- Replay buffer samples (index, priority, data index) by SumTree get method.
- From data index, get experiences including state, action, reward, next state.
- Pass index and priority to update method of SumTree and updated.
- PrioritizedReplay relies on SumTree and it contains priorities.

p = p_i = tree[index]
sampling_prob = P(i) = p_i^alpha / sum of p_i^alpha
tree.total() = sum of p_i^alpha

In the early training, just adding max_priority 1 to sum tree. When a prioritized replay memory contians some amount of
data, it starts updating priority, calculating loss, and updating model.
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
from collections import deque, namedtuple
import random
import time
import pickle

ENV = 'SpaceInvaders-v0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
PrioritizedExperience = namedtuple('Experience',
                                   ['state', 'action', 'reward', 'next_state', 'done', 'priority', 'index'])

# Parameter

# Deep Q Network
EPISODE = 10000
EPSILON_DECAY_RATE = 0.99999
STACK_FRAME = 3
# LR = 1e-3
LR = 1e-4
RENDER = True
TAU = 1e-3
DELAY_TRAINING = 1000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY_STEP = 20000
# BATCH_SIZE = 32
BATCH_SIZE = 64
MAXLEN = 100
SCORE = '../object/duel_ddqn_per_space_invaders_score.pkl'
MODEL = '../model/duel_ddqn_per_space_invaders_target_model.pth'

# experience replay
MEMORY_SIZE = 100000
BETA_RATE_PER = 0.9999999
ALPHA_PER = 0.6
BETA_PER = 0.4
EPSILON_PER = 0.01
# EPSILON_PER = 1e-6


class SumTree:
    """

    """
    def __init__(self, capacity):
        """
        In calling get method, pending_idx contains a new index retrieved. capacity of sum tree is memory_size of
        ExperienceReplay, which is the size of replay memory.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # print(f'len(self.tree): {len(self.tree)}')
        self.pending_idx = set()
        # write tracks the current position index to insert data. If write exceeds memory_size, goes back to 0
        self.write = 0
        # data has the same size as experience replay memory size, but what is this for?
        # self.data = np.zeros(capacity, dtype=object)
        # n_entries tracks currently how many data the replay memory contains
        self.n_entries = 0

    def get(self, s):
        """
        argument
        s is sampled from uniform distributions. data_index is used to sample experiences as batch

        return

        """
        index = self._retrieve(0, s)
        # Why is this equation to get data_index
        data_index = index - self.capacity + 1
        self.pending_idx.add(index)
        priority = self.tree[index]
        return index, priority, data_index

    def _retrieve(self, index, s):
        """
        _retrieve recursively finds one of leaf values. Argument index starts from 0 and is recursively updated with
        index in the tree, and eventually the index is returned as index. s is sampled from uniform distribution. If s
        is smaller than left child, go left down with s. But if s is bigger than the left child, go right down with
        value s - left child.
        """
        left = 2 * index + 1
        right = left + 1

        # Return one of the indices at the tree bottom leaf level
        if left >= len(self.tree):
            return index

        if s <= self.tree[left]:
            # print(f'left: {left}, s: {s}')
            return self._retrieve(left, s)
        else:
            # print(f'right: {right}, s - tree[left]: {s - self.tree[left]}')
            return self._retrieve(right, s - self.tree[left])

    def update(self, index, priority):
        """
        Update priority at the argument index position in sum tree with the argument priority.
        """
        # If this index is not in pending_idx?
        if index not in self.pending_idx:
            return
        self.pending_idx.remove(index)
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def _propagate(self, index, change):
        """
        index 0 1 2 3 4
              0
            /  \
           1    2
         / \   / \
        3  4  5   6
        If index is 1, (1 - 1) // 2 = 0
        If index is 6, (6 - 1) // 2 = 5 // 2 = 2
        """
        # This equation gets the index of a parent of a child
        parent = (index - 1) // 2
        # Because a child's value changed by the amount 'change', the parent's value also needs to be increased by the
        # amount 'change'.
        self.tree[parent] += change
        # As long as parent is not root, recursively update parents' value from bottom until the top root
        if parent != 0:
            self._propagate(parent, change)

    def add(self, p, data):
        """
        p is current max_priority, data is none. Store priority and sample
        """
        # If write is 0, index is the last index of replay memory
        index = self.write + self.capacity - 1
        self.pending_idx.add(index)

        # Maybe sum tree also contains experience data in array. Store sample?
        # self.data[self.write] = data
        # Store priority
        self.update(index, p)

        # Count up write
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def total(self):
        return self.tree[0]


class PrioritizedReplay:
    def __init__(self, memory_size,
                 alpha_per=ALPHA_PER, beta_per=BETA_PER, beta_rate_per=BETA_RATE_PER,
                 epsilon_per=EPSILON_PER,
                 batch_size=BATCH_SIZE,
                 history_length=STACK_FRAME):
        self.memory_size = memory_size
        self.tree = SumTree(memory_size)
        self.alpha_per = alpha_per
        self.beta_per = beta_per
        self.beta_rate_per = beta_rate_per
        self.epsilon_per = epsilon_per
        self.batch_size = batch_size
        self.max_priority = 1
        self._size = 0
        # pos is current position to add a new experience to replay memory
        # If pos is bigger than memory_size, pos goes back to 0
        self.pos = 0
        # history_length is so called number of stacking images to capture movement
        self.history_length = history_length

        # Experience
        self.state_buffer = [(np.zeros((84, 84), dtype=np.float32)) for _ in range(memory_size)]
        self.action_buffer = np.zeros(memory_size)
        self.reward_buffer = np.zeros(memory_size)
        self.done_buffer = np.zeros(memory_size)
        # self.state_buffer = []
        # self.action_buffer = []
        # self.reward_buffer = []
        # self.done_buffer = []
        # ?
        self.available_sample = 0

    def sample(self, batch_size):
        """
        Cut a line of length tree.total() into segments of number batch_size
        e.g. tree.total()=10 and batch_size=5 -> 0-2, 2-4, 4-6, 6-8, 8-10
        """

        sampled_data = []

        # Sample from nonoverlapping uniform distributions
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            # I don't understand this uniform distribution sampling
            s = random.uniform(a, b)
            # print(f'a: {a}\tb: {b}\ts: {s:.1f}')

            index, priority, data_index = self.tree.get(s)
            # print(f'index: {index}\tpriority: {priority}\tdata_index: {data_index}')
            # print()

            sampling_prob = priority / self.tree.total()

            # Use data_index to get the end of the current state
            # The idea of start and end comes from stacking several states to capture movement
            if not self.valid_index(data_index):
                # print('Invalid data index')
                continue

            state_start_index = data_index - self.history_length + 1
            state_end_index = data_index
            next_state_start_index = state_start_index + 1
            next_state_end_index = state_end_index + 1

            # print(f'state_start_index: {state_start_index}\t'
            #       f'state_end_index: {state_end_index}\t'
            #       f'next_state_start_index: {next_state_start_index}\t'
            #       f'next_state_end_index: {next_state_end_index}')

            state = [self.state_buffer[i] for i in range(state_start_index, state_end_index + 1)]
            action = self.action_buffer[state_end_index]
            # reward = [self.reward_buffer[i] for i in range(state_end_index, state_end_index + 1)]
            reward = self.reward_buffer[state_end_index]
            next_state = [self.state_buffer[i] for i in range(next_state_start_index, next_state_end_index + 1)]
            # done = [self.done_buffer[i] for i in range(state_end_index, state_end_index + 1)]
            done = self.reward_buffer[state_end_index]

            # From list to numpy (batch_size, hight, width)
            state = np.array(state)
            next_state = np.array(next_state)

            # print(f'{type(state)}, {type(action)}, {type(reward)}, {type(next_state)}, {type(done)}, '
            #       f'{type(priority)}, {type(index)}')

            experience = PrioritizedExperience(
                state=state, action=action, reward=reward, next_state=next_state, done=done,
                priority=sampling_prob, index=index
            )

            sampled_data.append(experience)

        # It's said this should rarely happen, but what is this?
        while len(sampled_data) < batch_size:
            # print('This should rarely happen')
            sampled_data.append(random.choice(sampled_data))

        # Convert list of namedtuples into namedtuple of numpy arrays
        # state, next_state: (batch_size, stack, height, width)
        # action, reward, done, priority, index: (batch_size,)
        sampled_data = zip(*sampled_data)
        sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        sampled_data = PrioritizedExperience(*sampled_data)

        # print(f'{type(sampled_data.state)}, {type(sampled_data.action)}, {type(sampled_data.reward)}, '
        #       f'{type(sampled_data.next_state)}, {type(sampled_data.done)}, {type(sampled_data.priority)}, '
        #       f'{type(sampled_data.index)}')
        # print(f'{sampled_data.state.shape}, {sampled_data.action.shape}, {sampled_data.reward.shape}, '
        #       f'{sampled_data.next_state.shape}, {sampled_data.done.shape}, {sampled_data.priority.shape}, '
        #       f'{sampled_data.index.shape}')

        # Anneal beta from initial beta_per to 1.0
        self._update_beta()

        return sampled_data

    def update_priorities(self, zip_index_priority):
        for index, priority in zip_index_priority:
            # max_priority starts from 1 and updated
            # print(f'index: {index}\tpriority: {priority}\tself.max_priority: {self.max_priority}')
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(index, priority)

    def _update_beta(self):
        """
        When sample a mini batch from buffer by sample method of PrioritizedReplay, anneal beta from initial beta to 1.0
        """
        self.beta_per = min(1.0, self.beta_per * self.beta_rate_per ** -1)
        return self.beta_per

    def append(self, experience_tuple):
        # Add a new experience to replay memory
        # print(f'len(self.state_buffer): {len(self.state_buffer)}\tself.pos: {self.pos}')
        # print(f'experience_tuple[0].cpu().numpy().shape: {experience_tuple[0].cpu().numpy().shape}\t'
        #       f'self.state_buffer[self.pos].shape: {self.state_buffer[self.pos].shape}')
        self.state_buffer[self.pos] = experience_tuple[0].cpu().numpy()
        self.action_buffer[self.pos] = experience_tuple[1]
        self.reward_buffer[self.pos] = experience_tuple[2]
        self.done_buffer[self.pos] = experience_tuple[3]

        # Update priority
        self.tree.add(self.max_priority, None)

        # Count up position index
        self.pos += 1
        if self.pos >= self.memory_size:
            self.pos = 0

        # Count up current replay memory size
        size = self._size
        size += 1
        self._size = min(self.memory_size, size)

        # What is available sample
        if self.available_sample + 1 < self.memory_size:
            self.available_sample += 1
        else:
            self.available_sample = self.memory_size - 1

    def valid_index(self, index):
        """
        Avoid index out of range. This can be caused by the stacked states for model input and n steps TD.
        """
        # When the replay memory is not yet filled up to memory_size
        # Make sure index is between 0 to current position of replay memory
        if index - self.history_length + 1 >= 0 and index + 1 < self.pos:
            return True

        # After the replay memory is filled up to memory size
        # If the index is beyond current position, the index must be within memory_size
        if index - self.history_length + 1 >= self.pos and index + 1 < self._size:
            return True

        return False

    def __len__(self):
        return self._size


class PERAgent:
    def __init__(self, env, device, online_model, target_model, action_algorithm,
                 tau, optimizer, gamma, memory, seed=0, online_network=None, target_network=None):
        self.online_network = online_network
        self.target_network = target_network
        self.num_action = env.action_space.n
        self.env = env
        self.device = device
        self.online_model = online_model
        self.target_model = target_model
        self.action_algorithm = action_algorithm
        self.tau = tau
        self.gamma = gamma
        self.optimizer = optimizer
        self.memory = memory

        # Seed
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def compute_loss(self, mini_batch):
        """
        Compute loss from q target and q estmate
        Argument
        mini_batch is a namedtuple PrioritizedExperience, each field contains numpy array
        """
        batch_size = len(mini_batch.action)
        # print(batch_size)

        # Convert numpy array to torch tensor
        state = torch.tensor(data=mini_batch.state, dtype=torch.float, device=self.device)
        action = torch.tensor(data=mini_batch.action, dtype=torch.long, device=self.device)
        reward = torch.tensor(data=mini_batch.reward, dtype=torch.float, device=self.device)
        next_state = torch.tensor(data=mini_batch.next_state, dtype=torch.float, device=self.device)
        done = torch.tensor(data=mini_batch.done, dtype=torch.long, device=self.device)

        # Convert torch.Size([batch_size]) to ([batch_size, 1])
        action = action.unsqueeze(1)
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)

        # print(f'{state.size()}, {action.size()}, {reward.size()}, {next_state.size()}, {done.size()}')

        # Double DQN
        # with torch.no_grad():
        action_indices = self.online_model(next_state).max(dim=1)[1]
        q_values = self.target_model(next_state).detach()
        max_q_values = q_values[np.arange(batch_size), action_indices].unsqueeze(1)

        # print(f'{self.target_model(next_state).size()}, '
        #       f'{self.online_model(next_state).size()}, '
        #       f'{self.online_model(state).size()} '
        #       f'{action.size()}')
        # print(f'{self.online_model(state)}, {action.unsqueeze(1).size()}')

        # Target
        q_target = reward + (self.gamma * max_q_values * (1 - done))

        # Estimate
        q_estimate = self.online_model(state).gather(dim=1, index=action)

        # Loss
        loss = q_target - q_estimate

        # print(f'{loss.size()}, {q_target.size()}, {q_estimate.size()}')

        return loss

    def prioritized_replay(self, loss, mini_batch):
        """
        Calculate priority from loss, update priority in sum tree, multiply loss by weight of improtance sampling.
        Argument
        loss is torch tensor size (batch_size, 1)
        index is a namedtuple PrioritizedExperience index field
        """

        # Update prioritized replay and sum tree
        priority = loss.abs().add(self.memory.epsilon_per).pow(self.memory.alpha_per)
        # (batch_size, 1) to (batch_size)
        priority = priority.squeeze()
        index = torch.tensor(data=mini_batch.index, dtype=torch.long, device=self.device)
        # print(f'priority: {priority.size()}\tindex: {index.size()}')
        self.memory.update_priorities(zip(index.cpu().detach().numpy(),
                                          priority.cpu().detach().numpy()))

        # Multiply loss by weight of importance sampling
        sampled_priority = torch.tensor(data=mini_batch.priority, dtype=torch.float, device=self.device)
        weight_per = sampled_priority.mul(sampled_priority.size(0)).add(self.memory.epsilon_per).pow(-self.memory.beta_per)
        weight_per = weight_per / weight_per.max()
        # (batch_size) to (batch_size, 1)
        weight_per = weight_per.unsqueeze(1)
        loss = loss.mul(weight_per)

        # print(f'weight_per: {weight_per.size()}, {loss.size()}')

        return loss

    def update_online_model(self, loss):
        """
        Apply gradient descent to update online model
        Argument
        loss is losses from mini batch
        """
        # Mean squared error
        # print(f'{loss.size()}, {loss}')
        loss = loss.pow(2).mul(0.5).mean()
        # print(f'{loss.size()}, {loss}')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self):
        """
        Update weight of neural network
        """
        return None

    def save_network(self, model_path, model):
        return None

    def random_play(self, sleep=0.05):
        self.env.reset()
        while True:
            self.env.render()
            action = np.random.randint(self.num_action)
            state, _, done, _ = self.env.step(action)
            time.sleep(sleep)
            if done:
                break

    def process_state(self, state):
        x = torch.tensor(data=state, dtype=torch.float, device=self.device)
        # Change color channel position from (210, 160, 3) to (1, 210, 160)
        x = x.permute(2, 0, 1)
        # From color to gray
        x = rgb_to_grayscale(x)
        # Resize from (1, 210, 160) to (1, 84, 84)
        x = Resize([84, 84])(x)
        # Reduce size 1 dimension
        x = x.squeeze(0)
        return x

    def stack_state(self, state1, state2, state3):
        x = torch.stack([state3, state2, state1])
        # Add batch size dimension
        x = x.unsqueeze(0)
        return x

    def soft_update_model(self):
        for target_param, online_param in zip(self.target_model.parameters(),
                                              self.online_model.parameters()):
            target_param.data.copy_(self.tau * online_param.data +
                                    (1.0 - self.tau) * target_param.data)


class EpsilonGreedyAlgorithm:
    def __init__(self, epsilon_start, epsilon_min,
                 epsilon_decay_rate=EPSILON_DECAY_RATE, epsilon_decay_step=None):
        self.epsilon = epsilon_start
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step

    def _epsilon_update(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))

        # Decay epsilon
        self._epsilon_update()

        return action


class DuelingQNetwork(nn.Module):
    def __init__(self, output_dim):
        super(DuelingQNetwork, self).__init__()

        # # in_channels=2 because 2 frames are stacked
        # # (80 + 2 * 0 - 1 * (6 - 1) - 1) / 2 + 1 = (80 - 6) / 2 + 1 = 74 / 2 + 1 = 38
        # # From (80 * 80 * 2) to (38 * 38 * 4)
        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2)
        # # (38 + 2 * 0 - 1 * (6 - 1) - 1) / 4 + 1 = (38 - 6) / 4 + 1 = 32 / 4 + 1 = 9
        # # From (38 * 38 * 4) to (9 * 9 * 16)
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=4)
        # self.size = 9 * 9 * 16
        # self.fc1 = nn.Linear(self.size, 256)
        # self.state_value = nn.Linear(256, 1)
        # self.action_advantage = nn.Linear(256, output_dim)

        # (84 + 2 * 0 - 1 * (8 - 1) - 1) / 4 + 1 = (84 - 8) / 2 + 1 = 76 / 2 + 1 = 38 + 1 = 39
        # From (84 * 84 * 3) to (39 * 39 * 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=0)
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

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = F.relu(x)
    #     x = x.view(-1, self.size)  # Flatten
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     # Action-advantage
    #     a = self.action_advantage(x)
    #     # State-value
    #     v = self.state_value(x).expand_as(a)
    #     # Action-value
    #     q = v + a - a.mean(1, keepdim=True).expand_as(a)
    #     return q

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # x = x.view(-1, self.size)  # Flatten
        x = x.view(-1, 3136)
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

    # Environment
    env = gym.make(ENV)
    print(f'Environment: {env}\nState: {env.observation_space}\nAction: {env.action_space.n}\n')

    # Model
    online_model = DuelingQNetwork(output_dim=env.action_space.n).to(device)
    target_model = DuelingQNetwork(output_dim=env.action_space.n).to(device)

    # Optimizer
    optimizer = optim.Adam(online_model.parameters(), lr=LR)

    # Epsilon greedy algorithm
    action_algorithm = EpsilonGreedyAlgorithm(epsilon_start=EPSILON_START,
                                              epsilon_min=EPSILON_MIN,
                                              epsilon_decay_rate=EPSILON_DECAY_RATE)

    # Replay memory
    memory = PrioritizedReplay(memory_size=MEMORY_SIZE)
    # memory = PrioritizedReplay(memory_size=1000)

    # Agent
    agent = PERAgent(env=env, device=device,
                     online_model=online_model, target_model=target_model,
                     action_algorithm=action_algorithm,
                     tau=TAU, gamma=GAMMA, optimizer=optimizer, memory=memory)

    # Random play
    # agent.random_play()

    # Initialization for training
    ma_reward_deque = deque(maxlen=MAXLEN)
    reward_list = []
    total_time_step = 0
    start_time = time.time()

    # Training
    # for i in range(EPISODE):
    for i in range(3):

        # Initialization for each episode
        state1 = env.reset()
        state2, _, _, _ = env.step(0)
        state3, _, _, _ = env.step(0)
        processed_state1 = agent.process_state(state1)
        processed_state2 = agent.process_state(state2)
        processed_state3 = agent.process_state(state3)
        stack_state = agent.stack_state(state1=processed_state1,
                                        state2=processed_state2,
                                        state3=processed_state3)
        total_reward = 0
        time_step = 0
        start_time_episode = time.time()

        while True:
        # while total_time_step < 20:

            if RENDER:
                env.render()
                # time.sleep(0.05)

            action = agent.action_algorithm.select_action(model=agent.online_model, state=stack_state)

            state, reward, done, _ = env.step(action)

            # Store experience
            agent.memory.append((processed_state1, action, reward, done))

            # When we have sufficient experience in replay memory, we start training
            if total_time_step > DELAY_TRAINING:
            # if total_time_step > 100:

                # Sample mini batch
                mini_batch = agent.memory.sample(batch_size=BATCH_SIZE)
                # mini_batch = agent.memory.sample(batch_size=2)

                # Calculate losses of each batch
                loss = agent.compute_loss(mini_batch=mini_batch)

                # Apply prioritized replay algorithm
                loss = agent.prioritized_replay(loss=loss, mini_batch=mini_batch)

                # Gradient descent to update weights of neural network
                agent.update_online_model(loss=loss)

                # Update target model with online mode
                agent.soft_update_model()

            processed_state1 = processed_state2
            processed_state2 = processed_state3
            processed_state3 = agent.process_state(state)
            next_stack_state = agent.stack_state(state1=processed_state1,
                                                 state2=processed_state2,
                                                 state3=processed_state3)

            total_reward += reward

            # Go to the next time step
            stack_state = next_stack_state
            total_time_step += 1
            time_step += 1

            # Monitor in each time step
            nl = '\n'
            # print(
                # f'Time step: {time_step}\t| '
                # f'Max priority: {agent.memory.max_priority:.4f}\t| '
                # f'Memory position: {agent.memory.pos}\t| '
                # f'Memory size: {len(agent.memory)}\t| '
                # f'Epsilon action: {agent.action_algorithm.epsilon:.2f}\t| '
                # f'Action buffer: {memory.action_buffer[:15]}\t'
                # f'Tree: {np.array_repr(memory.tree.tree[:10]).replace(nl, "")}'
            # )
            print(f'\rTime step per episode: {time_step}\t| '
                  f'Action: {action}\t| '
                  , end='')

            # break
            if done:
                break

        ma_reward_deque.append(total_reward)
        reward_list.append(total_reward)

        # Monitor in each episode
        # print(f'\rEpisode: {i}\t| Total time step: {total_time_step}\t| '
        #       f'Time step per episode: {time_step}\t| '
        #       f'Average score: {np.mean(ma_reward_deque):.2f}\t| '
        #       f'Current write position in memory: {memory.pos}')

        print(
            f'\nEpisode: {i:,}\t| '
            f'Average score: {np.mean(ma_reward_deque):.1f}\t| '
            f'Episode score: {total_reward:.1f}\t| '
            f'Epsilon action: {agent.action_algorithm.epsilon:.3f}\t| '
            f'Alpha PER: {agent.memory.alpha_per}\t| '
            f'Beta PER: {agent.memory.beta_per:.3f}\t| '
            f'SumTree total: {agent.memory.tree.total():,.1f}\t| '
            f'Max priority: {agent.memory.max_priority:,.1f}\t| '
            f'Time step per episode: {time_step:,}\t| '
            f'Total time step: {total_time_step:,}\t| '
            f'Memory position: {agent.memory.pos:,}\t| '
            f'Memory size: {len(agent.memory):,}\t| '
            f'Total elapsed time: {(time.time() - start_time) / 60:,.1f} min\t| '
            f'Elapsed time per episode: {(time.time() - start_time_episode):,.1f} sec\t| '
            f'Last action: {action}\t| '
        )

    # Save score
    # pickle.dump(reward_list, open(SCORE, 'wb'))
    # print('Saved score')

    # Save model
    # torch.save(agent.target_model.state_dict(), MODEL)
    # print('Saved model')


if __name__ == '__main__':
    main()
