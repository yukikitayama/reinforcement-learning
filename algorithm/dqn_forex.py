import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import time
from baakbrothers.ForexEnv import ForexEnv


# Parameter
EPISODE = 30000
MAX_SIZE = 10000
MONITOR_INTERVAL = 100
COPY_INTERVAL = 100
EPSILON_DECAY = 0.9999
MA_REWARD = 100
EPSILON_MIN = 0.1
LOG_DIR = '../tensorboard/dqn_forex'
LENGTH = 30
SEED = 0
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EP_MIN = 1
GAMMA = 0.99
BATCH_SIZE = 32


class DqnForex(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DqnForex, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.model(x)


class ExperienceReplay:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = {'state': [], 'action': [], 'reward': [],
                       'next_state': [], 'done': []}

    def store(self, experience):
        # Discard the oldest experience if the buffer is full
        if len(self.buffer['state']) >= self.max_size:
            for key in self.buffer.keys():
                self.buffer[key].pop(0)
        # Add new experience at the end
        for key, value in experience.items():
            self.buffer[key].append(value)

    def size(self):
        return len(self.buffer['state'])


def get_action(state, num_actions, model, epsilon):
    # Epsilon
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    # Greedy
    else:
        state = torch.tensor(state)
        return torch.argmax(model(state))


def update_model(model, target_model, memory, min_experience, batch_size,
                 gamma, num_actions, optimizer):
    # Do not update model when memory is small
    if memory.size() < min_experience:
        return 0

    # Mini batch
    index = np.random.choice(memory.size(), size=batch_size)
    states = torch.tensor([memory.buffer['state'][i] for i in index])
    actions = torch.tensor([memory.buffer['action'][i] for i in index])
    rewards = torch.tensor([memory.buffer['reward'][i] for i in index])
    next_states = torch.tensor([memory.buffer['next_state'][i] for i in index])
    dones = torch.tensor([memory.buffer['done'][i] for i in index])

    # Next Q values from target model
    next_action_values = torch.max(target_model(next_states), dim=1)
    next_action_values = torch.where(dones,
                                     rewards,
                                     rewards + gamma * next_action_values[0])
    # Block backpropagation of loss from affecting next state predictions
    next_action_values = next_action_values.detach()

    # Current Q values from training model
    action_values = model(states) * F.one_hot(actions, num_classes=num_actions)
    action_values = torch.sum(action_values, dim=1)

    # Loss function
    loss = torch.mean(torch.square(next_action_values - action_values))

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def update_target(model, target_model):
    model.load_state_dict(target_model.state_dict())
    return model


def main():
    # TensorBoard
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Environment
    env = ForexEnv(length=LENGTH)
    num_states = env.num_states
    num_actions = env.num_actions
    print(f'USD/JPY forex environment:\n'
          f'State: {num_states}\n'
          f'Action: {num_actions}')

    # Seed
    env.set_seed(SEED)
    torch.manual_seed(SEED)

    # Model
    model = DqnForex(num_states=num_states, num_actions=num_actions)
    target_model = DqnForex(num_states=num_states, num_actions=num_actions)

    # Optimizer
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Experience replay
    memory = ExperienceReplay(max_size=MAX_SIZE)

    # Training
    epsilon = EPSILON_START
    ep_reward = []
    avg_reward = []
    start_time = time.time()
    epsilons = []
    for ep in range(EPISODE):
        total_reward = 0
        state = env.reset()
        t = 0

        while True:

            action = get_action(state, num_actions, model, epsilon)

            next_state, reward, done = env.step(action)
            reward = float(reward)

            total_reward += reward

            if done:
                break

            memory.store({'state': state, 'action': action, 'reward': reward,
                          'next_state': next_state, 'done': done})

            update_model(model=model, target_model=target_model, memory=memory,
                         min_experience=EP_MIN, batch_size=BATCH_SIZE,
                         gamma=GAMMA, num_actions=num_actions,
                         optimizer=optimizer)

            t += 1

            state = next_state

            if t % COPY_INTERVAL == 0:
                target_model = update_target(model, target_model)

        epsilons.append(epsilon)
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

        ep_reward.append(total_reward)
        ma_reward = np.mean(ep_reward[-MA_REWARD:])
        avg_reward.append(ma_reward)

        if ep % MONITOR_INTERVAL == 0:
            print(f'Episode: {ep:,.0f}, ma reward: {ma_reward:,.1f}, '
                  f'ep reward: {total_reward:,.1f}, epsilon: {epsilon:,.3f}, '
                  f'minute: {((time.time() - start_time) / 60):,.1f}')

        # TensorBoard log
        writer.add_scalar('epsilon', epsilon, ep)
        writer.add_scalar('episode reward', total_reward, ep)
        writer.add_scalar('average reward', ma_reward, ep)


if __name__ == '__main__':
    main()
