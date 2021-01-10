import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SmoothL1Loss
import torch.optim as optim
from torchvision.transforms import Resize
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
import random
from collections import deque
import time
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# Parameter
# EPISODE = 1000
# EPISODE = 10000
EPISODE = 100
DELAY_TRAINING = 50000
MEMORY_SIZE = 500000
LR = 0.001
TAU = 0.08
GAMMA = 0.99
RENDER = True
# MODEL_01 = '../model/duel_ddqn_per_space_invaders_target_model_v2.pth'
# MODEL_02 = '../model/duel_ddqn_per_space_invaders_online_model_v2.pth'
MODEL_01 = '../model/duel_ddqn_per_space_invaders_target_model_v3.pth'
MODEL_02 = '../model/duel_ddqn_per_space_invaders_online_model_v3.pth'
# SCORE = '../object/duel_ddqn_per_space_invaders_score_v2.pkl'
SCORE = '../object/duel_ddqn_per_space_invaders_score_v3.pkl'
MAX_EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_MIN_ITER = 500000
BATCH_SIZE = 32
ALPHA_PER = 0.6
BETA_PER = 0.4
MIN_BETA = 0.4
MAX_BETA = 1.0
BETA_DECAY_ITERS = 500000
RESIZE = 84
MIN_PRIORITY = 0.01  # Epsilon of prioritized experience replay in p_i = |delta| + epsilon
NUM_FRAMES = 4
MAX_LEN = 100
ENV = 'SpaceInvaders-v0'


class Node:
    """
    Classmethod is callable from class name and method instead of making object first, but also callable from the
    object.
    self.value is sum of priorities, and self.index is data index.
    """
    def __init__(self, left, right, is_leaf=False, idx=None):
        """
        idx is only set for leaf nodes
        """
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = sum(node.value for node in (left, right) if node is not None)
        self.parent = None
        self.idx = idx
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(left=None, right=None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf

    def __str__(self):
        return str(self.left) + '-' + str(self.idx) + '-' + str(self.right)


def create_tree(input):
    """
    SumTree is initialized with zeros in each node value
    """
    nodes = [Node.create_leaf(value, index) for index, value in enumerate(input)]
    leaf_nodes = nodes

    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]

    return nodes[0], leaf_nodes


def retrieve(value, node):
    """
    Recursively find the leaf nodes. value - node.left.value because
    """
    if node.is_leaf:
        return node
    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)


def update_priority(node, new_value):
    """
    Updates priority in a specific node in SumTree identified by index. "change" is used to recursively update the
    priority in above (parent) nodes in SumTree structure.
    """
    # First, get the difference between the new priority value and the current priority value in the node to recursively
    # update the above (parent) node priorities
    change = new_value - node.value

    # Second, update the priority in the node
    node.value = new_value

    # Recursively update its parent
    propagate_changes(change, node.parent)


def propagate_changes(change, node):
    # Update priority
    node.value += change
    # default value of parent is None, but as long as left or right has something, self is set
    # So the below code recursively update until it hits the root of SumTree, because SumTree root parent is None
    if node.parent is not None:
        propagate_changes(change, node.parent)


class PrioritizedReplay:
    """
    Index of self.leaf_nodes and the index of each experience buffer are linked by self.write. Use self.write to access
    a specific experience from these buffers and a specific priority from SumTree.
    self.available_samples tells us how many experience we have in buffer, which we need to get weights of importance
    sampling to multiply N to P(i).
    """
    def __init__(self, memory_size, alpha=ALPHA_PER, beta=BETA_PER,
                 min_priority=MIN_PRIORITY):
        self.memory_size = memory_size
        self.write = 0
        self.base_node, self.leaf_nodes = create_tree([0 for _ in range(memory_size)])
        self.state_buffer = [(np.zeros((RESIZE, RESIZE), dtype=np.float32)) for _ in range(memory_size)]
        self.action_buffer = np.zeros(memory_size)
        self.reward_buffer = np.zeros(memory_size)
        self.done_buffer = np.zeros(memory_size)
        self.alpha = alpha
        self.beta = beta
        self.min_priority = min_priority
        self.available_samples = 0  # Controls index out of bound

    def append(self, experience, priority):
        # Update replay memory
        self.state_buffer[self.write] = experience[0]
        self.action_buffer[self.write] = experience[1]
        self.reward_buffer[self.write] = experience[2]
        self.done_buffer[self.write] = experience[3]

        # Update priority
        self.update(self.write, priority)

        # Update index
        self.write += 1
        # When self.write reaches the end of buffer, go back to the beginning
        if self.write >= self.memory_size:
            self.write = 0

        # TODO: clarify why we need available_samples
        if self.available_samples + 1 < self.memory_size:
            self.available_samples += 1
        else:
            self.available_samples = self.memory_size - 1

    def update(self, index, priority):
        update_priority(self.leaf_nodes[index], self.adjust_priority(priority))

    def adjust_priority(self, priority):
        """
        p_i^alpha in (1) in Prioritized Experience Replay paper.
        The exponent alpha determines how much prioritization is used
        np.power(2, 3) = 8 because 2^3
        """
        adjusted_priority = np.power(priority + self.min_priority, self.alpha)

        # print(f'priority: {priority}, adjusted_priority: {adjusted_priority:.3f}')

        return adjusted_priority

    def sample(self, batch_size):

        # Sample data index and priority
        indices = []
        weights = []

        sample_no = 0
        while sample_no < batch_size:
            # TODO: clarify this uniform random sampling
            # print(f'self.base_node.value: {self.base_node.value}')
            sample_val = np.random.uniform(0, self.base_node.value)
            # From SumTree root recursively goes down to find leaf nodes
            samp_node = retrieve(sample_val, self.base_node)

            # TODO: Clarify this, though I think it controls index out of bound
            if NUM_FRAMES - 1 < samp_node.idx < self.available_samples - 1:
                indices.append(samp_node.idx)
                # p = P(i) = p_i^alpha / sum of p_k^alpha
                p = samp_node.value / self.base_node.value
                # N * P(i) in an inverse form in 3.4 Annealing the bias in the paper
                weights.append((self.available_samples + 1) * p)
                sample_no += 1

        weights = np.array(weights)
        # w_i = (1/N * 1/P(i))^beta in an inverse form
        weights = np.power(weights, -self.beta)
        # Normalize weights by 1 / max w_i
        weights = weights / np.max(weights)

        # Sample experiences
        states = np.zeros((batch_size, RESIZE, RESIZE, NUM_FRAMES), dtype=np.float32)
        next_states = np.zeros((batch_size, RESIZE, RESIZE, NUM_FRAMES), dtype=np.float32)
        actions = []
        rewards = []
        dones = []
        for i, index in enumerate(indices):
            # print(f'index: {index}')
            for j in range(NUM_FRAMES):
                states[i, :, :, j] = self.state_buffer[index + j - NUM_FRAMES + 1]
                next_states[i, :, :, j] = self.state_buffer[index + j - NUM_FRAMES + 2]
            actions.append(self.action_buffer[index])
            rewards.append(self.reward_buffer[index])
            dones.append(self.done_buffer[index])

        # print(f'states: {states.shape}, actions: {np.array(actions).shape}, '
        #       f'rewards: {np.array(rewards).shape}')
        return states, np.array(actions), np.array(rewards), next_states, np.array(dones), indices, weights


class Agent:
    def __init__(self, num_actions, online_model, target_model, optimizer, tau=TAU, gamma=GAMMA, device=device):
        self.num_actions = num_actions
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.tau = tau
        self.gamma = gamma
        self.device = device

    def get_per_error(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(data=states, dtype=torch.float, device=self.device)
        actions = torch.tensor(data=actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(data=rewards, dtype=torch.float, device=self.device)
        next_states = torch.tensor(data=next_states, dtype=torch.float, device=self.device)
        dones = torch.tensor(data=dones, dtype=torch.long, device=self.device)

        # From (batch_size, height, width, stack frames) to (batch_size, stack frames, height, width)
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)
        # Convert torch.Size([batch_size]) to ([batch_size, 1])
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        batch_size = len(actions)

        # print(f'states: {states.shape}, actions: {actions.shape}, rewards: {rewards.shape}, '
        #       f'next_states: {next_states.shape}, dones: {dones.shape}')

        # Predict Q(s,a) given the batch of states
        # prim_qt = self.online_model(states)

        # Predict Q(s',a') from the evaluation network
        # prim_qtp1 = self.online_model(next_states)

        # Update one index corresponding to the max action
        # target_q = prim_qt.detach().cpu().numpy()
        # target_q = prim_qt

        # The action selection from the online network
        # prim_action_tp1 = prim_qtp1.max(dim=1)[1]

        # The q value for the prim_action_tp1 from the target network
        # q_from_target = self.target_model(next_states)

        # print(f'q_from_target: {q_from_target.shape}, prim_action_tp1: {prim_action_tp1.shape}')

        # Target of Q learning
        # updates = rewards + (1 - dones) * self.gamma * q_from_target[np.arange(batch_size), prim_action_tp1]

        # print(f'updates: {updates.shape}')
        # print(f'target_q: {target_q.shape}, actions: {actions.shape}, updates: {updates.shape}')
        # print(f'target_q: {target_q}, actions: {actions}, updates: {updates}')

        # target_q[:, actions] = updates

        # print(f'target_q: {target_q}, {target_q.size()}')
        # print(f'prim_qt: {prim_qt}, {prim_qt.size()}')
        # print(f'actions: {actions}, {actions.size()}')


        # states.shape[0] is batch size
        # Calculate loss of target Q and estimate Q
        # target_values = target_q.gather(dim=1, index=actions.unsqueeze(1))
        # estimate_values = prim_qt.gather(dim=1, index=actions.unsqueeze(1))
        # print(f'target_values: {target_values}, estimate_values: {estimate_values}')
        # target_q[i, actions[i]] - prim_qt[i, actions[i]] for i in range(states.shape[0])

        # Something like Huber loss
        # loss_fn = SmoothL1Loss(reduction='none')
        # error = loss_fn(target_values, estimate_values)
        # print(f'error: {error}')

        # Double DQN
        action_indices = self.online_model(next_states).max(dim=1)[1]
        q_values = self.target_model(next_states)
        max_q_values = q_values[np.arange(batch_size), action_indices].unsqueeze(1)

        # Target
        q_target = rewards + (self.gamma * max_q_values * (1 - dones))

        # Estimate
        q_estimate = self.online_model(states).gather(dim=1, index=actions)

        # Error
        error = q_target - q_estimate

        # return target_q, error
        return error

    def train(self, memory, batch_size):
        states, actions, rewards, next_states, dones, indices, weights = memory.sample(batch_size)

        # target_q, error = self.get_per_error(states, actions, rewards, next_states, dones)
        error = self.get_per_error(states, actions, rewards, next_states, dones)

        # print(f'target_q: {target_q}, error: {error.detach().cpu().numpy()}, indices: {indices}')

        for i in range(len(indices)):
            # print(f'indices[i]: {indices[i]}, error[i]: {error[i].detach().cpu().numpy()[0]}')
            error_abs = np.abs(error[i].detach().cpu().numpy()[0])
            memory.update(indices[i], error_abs)

        weights = torch.tensor(data=weights, dtype=torch.float, device=self.device)

        # print(f'error: {error}, weights: {weights}')

        error = error.mul(weights)

        loss = error.pow(2).mul(0.5).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def process_state(self, state):
        x = torch.tensor(data=state, dtype=torch.float, device=self.device)
        # Change color channel position from (210, 160, 3) to (1, 210, 160)
        x = x.permute(2, 0, 1)
        # From color to gray
        x = rgb_to_grayscale(x)
        # Resize from (1, 210, 160) to (1, 84, 84)
        x = Resize([RESIZE, RESIZE])(x)
        # Reduce size 1 dimension
        x = x.squeeze(0)
        # Normalize input 0 to i
        x = x.div(255)
        return x.detach().cpu().numpy()

    def process_state_stack(self, state_stack, state):
        state = torch.tensor(data=state, dtype=torch.float, device=self.device)
        state = state.unsqueeze(0)
        # print(f'state: {state.shape}')
        state_stack = torch.cat((state_stack, state))
        # print(f'state_stack: {state_stack.shape}')
        state_stack = state_stack[1:, :, :]
        # print(f'state_stack: {state_stack.shape}')
        return state_stack

    def choose_action(self, state, eps, step):
        if step < DELAY_TRAINING:
            return random.randint(0, self.num_actions - 1)
        else:
            if random.random() < eps:
                return random.randint(0, self.num_actions - 1)
            else:
                with torch.no_grad():
                    state = state.unsqueeze(0)
                    q_values = self.online_model(state).detach().cpu().numpy().squeeze()
                action = np.argmax(q_values)
                return action

    def update_network(self):
        for target_param, online_param in zip(self.target_model.parameters(),
                                              self.online_model.parameters()):
            target_param.data.copy_(self.tau * online_param.data +
                                    (1.0 - self.tau) * target_param.data)

    def huber_loss(self, loss):
        return 0.5 * loss ** 2 if abs(loss) < 1.0 else abs(loss) - 0.5


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

    # Environment
    env = gym.make(ENV)
    num_actions = env.action_space.n

    # Model
    online_model = DuelingQNetwork(output_dim=num_actions).to(device)
    target_model = DuelingQNetwork(output_dim=num_actions).to(device)

    # Optimizer
    optimizer = optim.Adam(online_model.parameters(), lr=LR)

    # Prioritized replay memory
    memory = PrioritizedReplay(memory_size=MEMORY_SIZE)

    # Agent
    agent = Agent(num_actions=num_actions,
                  online_model=online_model, target_model=target_model,
                  optimizer=optimizer)

    # Initialization for training
    eps = MAX_EPSILON
    steps = 0
    ma_reward_deque = deque(maxlen=MAX_LEN)
    reward_list = []
    start_time = time.time()

    # Training
    # for i in range(1):
    for i in range(EPISODE):

        # Initialization for episode
        state = env.reset()
        state = agent.process_state(state)
        # print(state.shape)
        state_stack = torch.tensor(
            data=np.repeat(state, NUM_FRAMES).reshape((NUM_FRAMES, RESIZE, RESIZE)),
            dtype=torch.float,
            device=device)
        # print(state_stack.shape)
        total_reward = 0
        avg_loss = 0
        time_step = 0
        start_time_episode = time.time()

        while True:
            if RENDER:
                env.render()
            action = agent.choose_action(state=state_stack, eps=eps, step=steps)
            # print(f'action: {action}')
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            next_state = agent.process_state(next_state)
            old_state_stack = state_stack
            # print(f'state_stack: {state_stack.shape}, next_state: {next_state.shape}')
            state_stack = agent.process_state_stack(state_stack=state_stack, state=next_state)
            # print(f'state_stack: {state_stack.shape}')

            if steps > DELAY_TRAINING:
            # if steps > 5:
                # print(f'before loss total: {memory.base_node.value}')
                loss = agent.train(memory=memory, batch_size=BATCH_SIZE)
                # print(f'after loss total: {memory.base_node.value}')

                agent.update_network()

                # print(f'states: {old_state_stack.detach().cpu().numpy().reshape((1, RESIZE, RESIZE, NUM_FRAMES)).shape}, '
                #       f'actions: {np.array([action]).shape}, '
                #       f'rewards: {np.array([reward]).shape}, '
                #       f'next_states: {state_stack.detach().cpu().numpy().reshape((1, RESIZE, RESIZE, NUM_FRAMES)).shape}, '
                #       f'dones: {np.array([done])}')

                error = agent.get_per_error(states=old_state_stack.detach().cpu().numpy().reshape((1, RESIZE, RESIZE, NUM_FRAMES)),
                                            actions=np.array([action]),
                                            rewards=np.array([reward]),
                                            next_states=state_stack.detach().cpu().numpy().reshape((1, RESIZE, RESIZE, NUM_FRAMES)),
                                            dones=np.array([done]))

                # print(f'error: {error.detach().cpu().numpy()[0][0]}')
                error_abs = np.abs(error.detach().cpu().numpy()[0][0])

                # Store in memory
                memory.append((next_state, action, reward, done), error_abs)

            else:
                # TODO: Clarify this loss = -1
                loss = -1
                memory.append((next_state, action, reward, done), reward)

            avg_loss += loss

            if steps > DELAY_TRAINING:
                eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * \
                      (MAX_EPSILON - MIN_EPSILON) if steps < EPSILON_MIN_ITER else \
                    MIN_EPSILON
                beta = MIN_BETA + ((steps - DELAY_TRAINING) / BETA_DECAY_ITERS) * \
                       (MAX_BETA - MIN_BETA) if steps < BETA_DECAY_ITERS else \
                    MAX_BETA
                memory.beta = beta


            # print(f'steps: {steps}\t| '
            #       f'reward: {reward}\t| '
            #       f'memory.base_node.value: {memory.base_node.value:,.3f}\t| '
            #       f'memory.write: {memory.write}\t| '
            #       f'memory.available_samples: {memory.available_samples}\t| ')

            steps += 1
            time_step += 1

            # if steps > 10:
            if done:
                break
            # break

        ma_reward_deque.append(total_reward)
        reward_list.append(total_reward)

        print(
            # f'\nEpisode: {i:,}\t| '
            f'Episode: {i:,}\t| '
            f'Average score: {np.mean(ma_reward_deque):.1f}\t| '
            f'Episode score: {total_reward:.1f}\t| '
            f'Epsilon action: {eps:.3f}\t| '
            f'Alpha PER: {memory.alpha}\t| '
            f'Beta PER: {memory.beta:.3f}\t| '
            f'SumTree total: {memory.base_node.value:,.1f}\t| '
            # f'Max priority: {agent.memory.max_priority:,.1f}\t| '
            f'Time step per episode: {time_step:,}\t| '
            f'Total time step: {steps:,}\t| '
            f'Memory position: {memory.write:,}\t| '
            f'Memory size: {memory.available_samples:,}\t| '
            f'Total elapsed time: {(time.time() - start_time) / 60:,.1f} min\t| '
            f'Elapsed time per episode: {(time.time() - start_time_episode):,.1f} sec\t| '
            f'Last action: {action}\t| '
        )

    # Save score
    pickle.dump(reward_list, open(SCORE, 'wb'))
    print('Saved score')

    # Save target model
    torch.save(agent.target_model.state_dict(), MODEL_01)
    print('Saved target model')

    # Save online model
    torch.save(agent.online_model.state_dict(), MODEL_02)
    print('Saved online model')


if __name__ == '__main__':
    main()
