import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import pickle

# Parameter
# https://stackoverflow.com/questions/43894608/use-of-omp-num-threads-1-for-python-multiprocessing
os.environ['OMP_NUM_THREADS'] = "1"
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print(f'Device: {device}')
print(f'Number of CPUs: {mp.cpu_count()}')
NUM_WORKERS = mp.cpu_count()
# NUM_WORKERS = 2
print(f'Number of workers: {NUM_WORKERS}')
ENV = 'CartPole-v0'
env = gym.make(ENV)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
# Adam
LR = 0.0001
BETA_01 = 0.92
BETA_02 = 0.999
# Max number of episodes
MAX_EP = 4000
# MAX_EP = 10
# Frequency to update shared neural network
UPDATE_GLOBAL_ITER = 5
# UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
SEED = 0
REWARDS = '../object/a3c_cartpole_reward.pkl'
MODEL = '../model/a3c_cartpole.pt'
SAVEFIG_01 = '../image/a3c_cartpole.png'
print()


# Initialize weights in layers in neural network
def initialize_weights(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)


def np_to_tensor(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    tensor = torch.from_numpy(np_array)
    return tensor


# Record result when episode ends
def record(shared_episode_counter, shared_episode_reward, local_episode_reward, shared_rewards_queue, name):
    """
    global_episode: mp.Value integer
    global_episode: mp.Value float
    """
    # Increment shared episode counter
    with shared_episode_counter.get_lock():
        shared_episode_counter.value += 1

    # Update shared total rewards per episode as weighted average
    with shared_episode_reward.get_lock():
        if shared_episode_reward.value == 0.0:
            shared_episode_reward.value = local_episode_reward
        else:
            shared_episode_reward.value = (
                shared_episode_reward.value * 0.99 + local_episode_reward * 0.01
            )

    # Append local total reward to reward queue
    shared_rewards_queue.put(shared_episode_reward.value)

    print(f'{name} finished epeisode: {shared_episode_counter.value} with reward: {shared_episode_reward.value:,.1f}')


# Push local network to global network and then pull global network to local network
def push_and_pull(shared_optimizer,
                  local_network,
                  shared_network,
                  done,
                  next_state,
                  buffer_state,
                  buffer_action,
                  buffer_reward,
                  gamma):
    # State value of next state
    if done:
        next_state_value = 0.0
    else:
        # network returns [logits, values], so [-1] returns state value
        # forward(x).data.numpy() here returns data shape [1, 1], so to get a scalar [0, 0]
        next_state_value = local_network.forward(np_to_tensor(next_state[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    # Calculate discounted total reward at each time step
    for r in buffer_reward[::-1]:
        next_state_value = r + gamma * next_state_value
        buffer_v_target.append(next_state_value)
        # .reverse() updates the existing list
    buffer_v_target.reverse()

    # Loss is calculated from local neural network
    # In loss_func, state buffer shape (n, state dim), action buffer shape (n,), reward buffer shape (n, 1)
    loss = local_network.loss_func(
        # buffer_state is list of numpy array, so np.vstack(buffer_state) returns shape (n, state dimensions)
        np_to_tensor(np.vstack(buffer_state)),
        # if else is just in case. It should be int64 in CartPole
        np_to_tensor(np.array(buffer_action), dtype=np.int64) if buffer_action[0].dtype == np.int64 else np_to_tensor(np.vstack(buffer_action)),
        np_to_tensor(np.array(buffer_v_target)[:, None])
    )

    # Push
    # Update shared neural network parameters
    shared_optimizer.zero_grad()
    loss.backward()
    for lp, gp in zip(local_network.parameters(), shared_network.parameters()):
        # .grad is not writable, but ._grad is writable
        gp._grad = lp.grad
    shared_optimizer.step()

    # Pull
    # Update local neural network parameters
    local_network.load_state_dict(shared_network.state_dict())


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


class Network(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Network, self).__init__()
        # State
        self.s_dim = s_dim
        # Action
        self.a_dim = a_dim
        # Policy
        self.pi1 = nn.Linear(s_dim, 256)
        self.pi2 = nn.Linear(256, a_dim)
        # State value function
        self.v1 = nn.Linear(s_dim, 256)
        self.v2 = nn.Linear(256, 1)
        # Initialize weights
        initialize_weights([self.pi1, self.pi2, self.v1, self.v2])
        # Creates a categorical distribution parameterized by probabilities here
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        """
        x is state tensor.
        """
        # Logits for policy
        pi1 = torch.relu(self.pi1(x))
        logits = self.pi2(pi1)
        # State value function
        v1 = torch.relu(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, state):
        """
        s is state tensor
        """
        # Use eval() because it's inference
        self.eval()
        logits, _ = self.forward(state)
        # logits has size ([n, action dimensions])
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(probs=prob)
        # m = self.distribution(prob)
        # numpy() returns shape (1,) so numpy()[0] returns a scalar
        action = m.sample().numpy()[0]
        return action

    def loss_func(self, state, action, target_values):
        """
        s is state buffer shape (n, state dimensions)
        a is action buffer shape (n,)
        v_t is target state value buffer shape (n, 1)
        """
        # print(f'loss_func: state.size(): {state.size()}, action.size(): {action.size()}, target_values.size(): {target_values.size()}')

        self.train()
        logits, values = self.forward(state)

        # Value loss, td is an estimate of advantage function
        td = target_values - values
        c_loss = td.pow(2)

        # Policy loss
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs=probs)
        # m = self.distribution(probs)
        # From A3C paper, why detach()?
        exp_v = m.log_prob(action) * td.detach().squeeze()
        # Why negative?
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()

        # print(f'c_loss.size(): {c_loss.size()}, a_loss.size(): {a_loss.size()}, total_loss.size(): {total_loss.size()}')

        return total_loss


# Override the run method by subclassing mp.Process
class Worker(mp.Process):
    def __init__(self,
                 shared_network,
                 shared_optimizer,
                 shared_episode_counter,
                 shared_episode_reward,
                 shared_rewards_queue,
                 name):
        super(Worker, self).__init__()
        # Shared elements
        self.shared_network = shared_network
        self.shared_optimizer = shared_optimizer
        self.shared_episode_counter = shared_episode_counter
        self.shared_episode_reward = shared_episode_reward
        self.shared_rewards_queue = shared_rewards_queue
        # Local elements
        self.local_network = Network(N_S, N_A)
        self.local_env = gym.make(ENV).unwrapped
        self.local_env.seed(name)
        self.name = f'worker_{"0" + str(name) if name < 10 else str(name)}'
        print(f'{self.name} is made in process ID: {os.getpid()} and parent ID: {os.getppid()}')

    def run(self):
        print(f'*** {self.name} started in process ID: {os.getpid()} and parent ID: {os.getppid()} ***')

        # Total time step, not episode
        total_time_step = 1

        # The number of episodes experienced in training is shared over workers
        # When total number of episodes experience by all the workers exceeds a threshold,
        # A3C training ends
        while self.shared_episode_counter.value < MAX_EP:

            # Initialize
            state = self.local_env.reset()
            # States, actions, rewards
            buffer_state, buffer_action, buffer_reward = [], [], []
            total_rewards = 0.0

            while True:

                # Only print worker_00's interaction with environment
                if self.name == 'worker_01':
                    self.local_env.render()

                # Choose action
                # s[None, :] change s.shape (4,) to s[None, :].shape (1, 4)
                action = self.local_network.choose_action(np_to_tensor(state[None, :]))

                # Next state, reward, done
                next_state, reward, done, _ = self.local_env.step(action)

                if done:
                    # Reward is collected even after termination, but we shouldn't
                    # CartPole gains 1 every time step, so subtract 1 from total reward
                    reward = -1

                total_rewards += reward

                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward)

                # Update shared neural network only when episode ends or when at specified frequency
                if total_time_step % UPDATE_GLOBAL_ITER == 0 or done:

                    # Update shared neural network by local loss of gradients from local neural network
                    # Update local neural network parameters by the updated shared neural network parameters
                    push_and_pull(
                        shared_optimizer=self.shared_optimizer,
                        local_network=self.local_network,
                        shared_network=self.shared_network,
                        done=done,
                        next_state=next_state,
                        buffer_state=buffer_state,
                        buffer_action=buffer_action,
                        buffer_reward=buffer_reward,
                        gamma=GAMMA
                    )

                    # Every time updating, clear experience
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:
                        record(
                            shared_episode_counter=self.shared_episode_counter,
                            shared_episode_reward=self.shared_episode_reward,
                            local_episode_reward=total_rewards,
                            shared_rewards_queue=self.shared_rewards_queue,
                            name=self.name
                        )
                        break

                # Iterate to the next state
                state = next_state
                total_time_step += 1

        # Append None at the end of list of total rewards to call .join() to stop parallel programming
        self.shared_rewards_queue.put(None)


if __name__ == '__main__':

    # Parallel programming
    print('Start main')
    print(f'Module name: {__name__}')
    print(f'Parent process: {os.getppid()}')
    print(f'Process ID: {os.getpid()}')
    print()

    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env.seed(SEED)

    # Shared network
    shared_network = Network(N_S, N_A)

    # Share the shared network parameters in multiprocessing
    shared_network.share_memory()

    # Shared optimizer
    shared_optimizer = SharedAdam(shared_network.parameters(), lr=LR, betas=(BETA_01, BETA_02))

    # Initialize shared episode counter, shared ?, and shared ?
    shared_episode_counter = mp.Value('i', 0)  # i indicates a signed integer, mp.Value to store values in a shared memory
    shared_episode_reward = mp.Value('d', 0.)  # d indicates a double precision float, mp.Value to store values in a shared memory
    shared_rewards_queue = mp.Queue()  # Queue allows multiple processes to access the same queue object, meaning we don't need Lock

    # Parallel training
    # Make parallel agents by mp.Process subclass Worker
    print('Make parallel agents')
    workers = [
        Worker(
            shared_network,
            shared_optimizer,
            shared_episode_counter,
            shared_episode_reward,
            shared_rewards_queue,
            i)
        for i
        in range(NUM_WORKERS)
    ]

    # Start parallel training
    print()
    print('*** Start parallel training ***')
    [w.start() for w in workers]

    total_rewards = []
    while True:
        reward = shared_rewards_queue.get()
        if reward is not None:
            total_rewards.append(reward)
        else:
            break

    # End parallel training
    [w.join() for w in workers]
    print('*** End parallel training ***')
    print()

    # Store rewards
    pickle.dump(total_rewards, open(REWARDS, 'wb'))

    # Save model
    torch.save(shared_network.state_dict(), MODEL)

    # Show rewards
    plt.plot(total_rewards)
    plt.title(f'Asynchronous Advantage Actor Critic CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Moving average of episode rewards')
    plt.tight_layout()
    plt.savefig(SAVEFIG_01)
    plt.show()
