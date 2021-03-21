import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
# from torchvision.transforms import Resize
# from torchvision.transforms.functional import rgb_to_grayscale
import gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle

# https://stackoverflow.com/questions/43894608/use-of-omp-num-threads-1-for-python-multiprocessing
os.environ['OMP_NUM_THREADS'] = "1"

# Shared optimizer hyperparameters
LR = 0.0001
BETA_01 = 0.92
BETA_02 = 0.999

# Parallel
NUM_WORKERS = mp.cpu_count()
# NUM_WORKERS = 2  # Test
# UPDATE_SHARED_NETWORK_FREQ = 5
UPDATE_SHARED_NETWORK_FREQ = 50

# ENV = 'BreakoutNoFrameskip-v4'
# ENV = 'Breakout-v0'
ENV = 'SpaceInvaders-v0'
STACK_FRAME = 2
env = gym.make(ENV)
# print(f'Number of actions: {env.action_space.n}')
# ACTION_DIM = 4
ACTION_DIM = 6
MAX_EPISODE = 10000
# MAX_EPISODE = 10
# GAMMA = 0.99
GAMMA = 0.9
REWARDS = f'../object/a3c_{ENV}_reward.pkl'
MODEL = f'../model/a3c_{ENV}.pt'
SAVEFIG_01 = f'../image/a3c_{ENV}.png'
device = torch.device('cpu')


# Initialize weights in layers in neural network
def initialize_weights(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)


# Resize Atari state
def process_state(state):
    """
    Return a processed state size (1, 80, 80), (channel, height, width)
    """
    # print(f'Entered process_state')
    # Below does not work in parallel process
    # x = torch.tensor(data=state, dtype=torch.float)
    if state.dtype != np.float32:
        state = state.astype(np.float32)

    # From color to gray
    state = np.mean(state, axis=2)

    # Resize
    im = Image.fromarray(state)
    im = im.resize((80, 80))

    # From PIL Image to np array
    state = np.array(im)

    # Add batch size dimension
    state = state.reshape((1, 80, 80))
    # print(f'After PIL shape: {state.shape}')

    x = torch.from_numpy(state)
    # print(f'x.size(): {x.size()}, x.dtype: {x.dtype}')
    # Change color channel position from (210, 160, 3) to (1, 210, 160)
    # x = x.permute(2, 0, 1)
    # print(f'x.size() after permute: {x.size()}')
    # From color to gray
    # TODO: below is not working. Why?
    # print(f'type(x): {type(x)}')
    # x = x.unsqueeze(dim=0)
    # print(f'x.size(): {x.size()}')
    # x = rgb_to_grayscale(x)
    # x = x.mean(dim=1)
    # print(f'x.size(): {x.size()}')
    # Resize from (1, 210, 160) to (1, 80, 80)
    # x = Resize([80, 80])(x)
    # print('Exit process_state')
    return x


def stack_states(state, next_state):
    """
    Returns a stacked state size (1, 2, 80, 80), (batch size, stacked frame, height, width)
    """
    x = torch.stack([next_state, state], dim=1)
    return x


def push_and_pull(shared_network,
                  shared_optimizer,
                  local_network,
                  done,
                  stacked_state,
                  buffer_state,
                  buffer_action,
                  buffer_reward,
                  gamma):

    # At the end of the episode, we have no state value at the next state
    if done:
        next_state_value = 0.0
    else:
        # network returns [logits, values], so [-1] returns state value
        # forward(x).data.numpy() here returns data shape [1, 1], so to get a scalar [0, 0]
        next_state_value = local_network.forward(stacked_state)[-1].data.numpy()[0, 0]

    # Discounted total reward at each time step
    buffer_v_target = []
    for r in buffer_reward[::-1]:
        next_state_value = r + gamma * next_state_value
        buffer_v_target.append(next_state_value)
    buffer_v_target.reverse()

    # Loss
    loss = local_network.loss_function(
        state=torch.vstack(buffer_state),
        action=torch.tensor(buffer_action),
        target_values=torch.tensor(buffer_v_target).unsqueeze(dim=1)
    )

    # Update shared neural network (Push)
    shared_optimizer.zero_grad()
    loss.backward()
    for shared_params, local_params in zip(shared_network.parameters(), local_network.parameters()):
        # .grad is not writable, but ._grad is writable
        shared_params._grad = local_params.grad
    shared_optimizer.step()

    # Update local neural network (Pull)
    local_network.load_state_dict(shared_network.state_dict())


def record(shared_counter, shared_reward, shared_queue, local_reward, name):

    # Increment shared episode counter
    with shared_counter.get_lock():
        shared_counter.value += 1

    # Update shared total rewards per episode
    with shared_reward.get_lock():
        if shared_reward.value == 0:
            shared_reward.value = local_reward
        else:
            shared_reward.value = (
                shared_reward.value * 0.99 + local_reward * 0.01
            )

    # Append local total rewards per episode to reward queue
    shared_queue.put(shared_reward.value)

    print(f'{name} finished episode: {shared_counter.value} with reward: {shared_reward.value:,.1f}')


class Network(nn.Module):
    def __init__(self, stack_frame, action_dim):
        super(Network, self).__init__()
        self.action_dim = action_dim
        self.stack_frame = stack_frame
        # Policy
        # in_channels=2 because 2 frames are stacked
        # (80 + 2 * 0 - 1 * (6 - 1) - 1) / 2 + 1 = (80 - 6) / 2 + 1 = 74 / 2 + 1 = 38
        # From (80 * 80 * 2) to (38 * 38 * 4)
        self.policy_conv1 = nn.Conv2d(in_channels=stack_frame, out_channels=4, kernel_size=6, stride=2)
        # (38 + 2 * 0 - 1 * (6 - 1) - 1) / 4 + 1 = (38 - 6) / 4 + 1 = 32 / 4 + 1 = 9
        # From (38 * 38 * 4) to (9 * 9 * 16)
        self.policy_conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.policy_fc1 = nn.Linear(self.size, 256)
        self.policy_fc2 = nn.Linear(256, action_dim)
        # State value function
        self.value_conv1 = nn.Conv2d(in_channels=stack_frame, out_channels=4, kernel_size=6, stride=2)
        self.value_conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=4)
        self.value_fc1 = nn.Linear(self.size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        # Initialize weights
        initialize_weights([
            self.policy_conv1, self.policy_conv2, self.policy_fc1, self.policy_fc2,
            self.value_conv1, self.value_conv2, self.value_fc1, self.value_fc2
        ])
        # Creates a categorical distribution parameterized by probabilities here
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        # Logits for policy
        l = self.policy_conv1(x)
        l = F.relu(l)
        l = self.policy_conv2(l)
        l = F.relu(l)
        l = l.view(-1, self.size)  # Flatten
        l = self.policy_fc1(l)
        l = F.relu(l)
        logits = self.policy_fc2(l)
        # State value function
        v = self.value_conv1(x)
        v = F.relu(v)
        v = self.value_conv2(v)
        v = F.relu(v)
        v = v.view(-1, self.size)  # Flatten
        v = self.value_fc1(v)
        v = F.relu(v)
        values = self.value_fc2(v)
        return logits, values

    def choose_action(self, state):
        self.eval()
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=1).data
        print(f'probs: {probs}')
        m = self.distribution(probs=probs)
        action = m.sample().numpy()[0]
        return action

    def loss_function(self, state, action, target_values):
        """
        state size (batch size, stacked frames, height, width)
        action size (batch size)
        target_values size (batch size, 1)

        Returns a tensor size([]), one scaler in tensor
        """
        self.train()
        # logits (batch size, actions), values (batch size, 1)
        logits, values = self.forward(state)

        # Value loss
        td = target_values - values
        c_loss = td.pow(2)

        # Policy loss
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs=probs)
        # m.log_prob(action) size (batch size)
        # td size (batch size, 1) to td.detach().squeeze() size (batch size)
        # exp_v size (batch size)
        exp_v = m.log_prob(action) * td.detach().squeeze()
        a_loss = -exp_v

        # Total loss
        # c_loss size (batch size, 1), a_loss size (batch size), total_loss size ([])
        total_loss = (c_loss + a_loss).mean()
        return total_loss


# Adam optimizer shared by parallel workers
class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Worker(mp.Process):
    def __init__(self,
                 shared_network,
                 shared_optimizer,
                 shared_counter,
                 shared_reward,
                 shared_queue,
                 name):
        super(Worker, self).__init__()
        # Shared element
        self.shared_network = shared_network
        self.shared_optimizer = shared_optimizer
        self.shared_counter = shared_counter
        self.shared_reward = shared_reward
        self.shared_queue = shared_queue
        # Local element
        self.local_network = Network(STACK_FRAME, ACTION_DIM)
        # self.local_env = gym.make(ENV).unwrapped
        self.local_env = gym.make(ENV)
        self.local_env.seed(name)
        self.name = f'worker_{"0" + str(name) if name < 10 else str(name)}'
        print(f'{self.name} is made in process ID: {os.getpid()} and parent ID: {os.getppid()}')

    def run(self):
        print(f'*** {self.name} started in process ID: {os.getpid()} and parent ID: {os.getppid()} ***')

        # Initialize for training
        total_time_step = 1

        while self.shared_counter.value < MAX_EPISODE:

            # Initialize for each episode
            state1 = self.local_env.reset()
            state2, _, _, _ = self.local_env.step(0)
            state1 = process_state(state1)
            state2 = process_state(state2)
            stacked_state = stack_states(state=state1, next_state=state2)
            buffer_state = []
            buffer_action = []
            buffer_reward = []
            total_rewards = 0.0

            while True:

                # Only print one worker to check the interaction with environment
                if self.name == 'worker_00':
                    self.local_env.render()

                # Choose action
                action = self.local_network.choose_action(stacked_state)

                # Interact with the environment
                next_state, reward, done, _ = self.local_env.step(action)
                # print(f'{self.name} took action: {action} and received done: {done} at time step: {total_time_step}')

                # Collect experience
                total_rewards += reward
                buffer_state.append(stacked_state)
                buffer_action.append(action)
                buffer_reward.append(reward)

                # Update shared neural network
                if total_time_step % UPDATE_SHARED_NETWORK_FREQ == 0 or done:

                    push_and_pull(
                        shared_network=self.shared_network,
                        shared_optimizer=self.shared_optimizer,
                        local_network=self.local_network,
                        done=done,
                        stacked_state=stacked_state,
                        buffer_state=buffer_state,
                        buffer_action=buffer_action,
                        buffer_reward=buffer_reward,
                        gamma=GAMMA
                    )

                    # Clear experience buffers
                    buffer_state.clear()
                    buffer_action.clear()
                    buffer_reward.clear()

                    if done:

                        record(
                            shared_counter=self.shared_counter,
                            shared_reward=self.shared_reward,
                            shared_queue=self.shared_queue,
                            local_reward=total_rewards,
                            name=self.name
                        )

                        break

                # Iterate to the next state
                state1 = state2
                state2 = process_state(next_state)
                stacked_state = stack_states(state=state1, next_state=state2)

                total_time_step += 1

        # Append None at the end of the queue to call .join() and stop the parallel training
        self.shared_queue.put(None)


if __name__ == '__main__':

    # Check parallel state
    print('Start main')
    print(f'Number of CPUs of this machine: {mp.cpu_count()}')
    print(f'Number of workers set in this training: {NUM_WORKERS}')
    print(f'__name__: {__name__}')
    print(f'Parent process ID: {os.getppid()}')
    print(f'Current process ID: {os.getpid()}')
    print(f'mp.get_all_sharing_strategies(): {mp.get_all_sharing_strategies()}')
    print(f'mp.get_sharing_strategy(): {mp.get_sharing_strategy()}')
    # https://github.com/pytorch/pytorch/issues/2496
    mp.set_start_method('spawn')
    print('Set start method as spawn')
    print()

    # Shared network
    shared_network = Network(stack_frame=STACK_FRAME, action_dim=ACTION_DIM)
    shared_network.share_memory()

    # Shared optimizer
    shared_optimizer = SharedAdam(shared_network.parameters(), lr=LR, betas=(BETA_01, BETA_02))

    # Shared episode counter, episde total reward, queue to store total rewards from each worker
    shared_counter = mp.Value('i', 0)
    shared_reward = mp.Value('d', 0.0)
    shared_queue = mp.Queue()

    # Parallel workers
    print('Make parallel workers')
    workers = [
        Worker(
            shared_network=shared_network,
            shared_optimizer=shared_optimizer,
            shared_counter=shared_counter,
            shared_reward=shared_reward,
            shared_queue=shared_queue,
            name=i
        )
        for i in range(NUM_WORKERS)
    ]

    # Start parallel training
    print()
    print('*** Start parallel training ***')
    [worker.start() for worker in workers]

    total_rewards = []
    while True:
        reward = shared_queue.get()
        if reward is not None:
            total_rewards.append(reward)
        else:
            break

    # End parallel training
    [worker.join() for worker in workers]
    print('*** End parallel training ***')
    print()

    # Store rewards
    pickle.dump(total_rewards, open(REWARDS, 'wb'))

    # Save model
    torch.save(shared_network.state_dict(), MODEL)

    # Show rewards
    plt.plot(total_rewards)
    plt.title(f'Asynchronous Advantage Actor Critic {ENV}')
    plt.xlabel('Episode')
    plt.ylabel('Moving average of episode rewards')
    plt.tight_layout()
    plt.savefig(SAVEFIG_01)
    plt.show()
