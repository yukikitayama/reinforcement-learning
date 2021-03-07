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

# I don't know what it is, but without it, it works
# os.environ['OMP_NUM_THREADS'] = "1"

# Parameter
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
MAX_EP = 3000
# MAX_EP = 10
# Frequency to update shared neural network
UPDATE_GLOBAL_ITER = 5
# UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
REWARDS = '../object/a3c_cartpole_reward.pkl'
MODEL = '../model/a3c_cartpole.pt'

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
def record(global_episode, global_episode_reward, local_episode_reward, res_queue, name):
    """
    global_episode: mp.Value integer
    global_episode: mp.Value float
    """
    # Increment shared episode counter
    with global_episode.get_lock():
        global_episode.value += 1

    # Update shared total rewards per episode as weighted average
    with global_episode_reward.get_lock():
        if global_episode_reward.value == 0.0:
            global_episode_reward.value = local_episode_reward
        else:
            global_episode_reward.value = (
                global_episode_reward.value * 0.99 + local_episode_reward * 0.01
            )

    # Store current total rewards
    res_queue.put(global_episode_reward.value)

    # Append local total reward to reward queue
    res_queue.put(global_episode_reward.value)

    print(f'{name} finished epeisode: {global_episode.value} with reward: {global_episode_reward.value}')


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
        v_s_ = 0.0
    else:
        # network returns [logits, values], so [-1] returns state value
        # Why [0, 0]?
        v_s_ = local_network.forward(np_to_tensor(next_state[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    # Calculate discounted total reward at each time step
    for r in buffer_reward[::-1]:
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
        # .reverse() updates the existing list
    buffer_v_target.reverse()

    # Loss is calculated from local neural network
    # In loss_func, state buffer shape (n, state dim), action buffer shape (n,), reward buffer shape (n, 1)
    loss = local_network.loss_func(
        # buffer_state is list of numpy array, so np.vstack(buffer_state) returns shape (n, state dimensions)
        np_to_tensor(np.vstack(buffer_state)),
        # if else is just in case. It should be int64 in CartPole
        np_to_tensor(np.array(buffer_action), dtype=np.int64) if buffer_action[0].dtype == np.int64 else np_to_tensor(np.vstack(buffer_action)),
        np_to_tensor(np.array(buffer_reward)[:, None])
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
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        # State value function
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        # Initialize weights
        initialize_weights([self.pi1, self.pi2, self.v1, self.v2])
        # Creates a categorical distribution parameterized by probabilities here
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        """
        x is state tensor.
        """
        # Logits for policy
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        # State value function
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        """
        s is state tensor
        """
        # Use eval() because it's inference
        self.eval()
        logits, _ = self.forward(s)
        # logits has size ([n, action dimensions])
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(probs=prob)
        # numpy() returns shape (1,) so numpy()[0] returns a scalar
        action = m.sample().numpy()[0]
        return action

    def loss_func(self, s, a, v_t):
        """
        s is state buffer shape (n, state dimensions)
        a is action buffer shape (n,)
        v_t is target state value buffer shape (n, 1)
        """
        self.train()
        logits, values = self.forward(s)

        # Value loss, td is an estimate of advantage function
        td = v_t - values
        c_loss = td.pow(2)

        # Policy loss
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs=probs)
        # From A3C paper, why detach()?
        exp_v = m.log_prob(a) * td.detach().squeeze()
        # Why negative?
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


# Override the run method by subclassing mp.Process
class Worker(mp.Process):
    def __init__(self, global_network, global_optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = f'worker_{"0" + str(name) if name < 10 else str(name)}'
        print(f'{self.name} is made in process ID: {os.getpid()} and parent ID: {os.getppid()}')
        # Number of episode shared
        self.g_ep = global_ep
        self.g_ep_r = global_ep_r
        self.res_queue = res_queue
        # Shared neural network
        self.gnet = global_network
        # Shared optimizer
        self.opt = global_optimizer
        # Local neural network
        self.lnet = Network(N_S, N_A)
        # Local environment of each worker
        self.env = gym.make(ENV).unwrapped

    def run(self):
        print(f'*** {self.name} started in process ID: {os.getpid()} and parent ID: {os.getppid()} ***')

        # Total time step, not episode
        total_step = 1

        # The number of episodes experienced in training is shared over workers
        # When total number of episodes experience by all the workers exceeds a threshold,
        # A3C training ends
        while self.g_ep.value < MAX_EP:

            # Initialize
            s = self.env.reset()
            # States, actions, rewards
            buffer_s, buffer_a, buffer_r = [], [], []
            # Total rewards
            ep_r = 0.0

            while True:

                # Only print worker_00's interaction with environment
                if self.name == 'worker_00':
                    self.env.render()

                # Choose action
                # s[None, :] change s.shape (4,) to s[None, :].shape (1, 4)
                a = self.lnet.choose_action(np_to_tensor(s[None, :]))

                # Next state, reward, done
                s_, r, done, _ = self.env.step(a)

                if done:
                    # Reward is collected even after termination, but we shouldn't
                    # CartPole gains 1 every time step, so subtract 1 from total reward
                    r = -1

                ep_r += r

                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                # Update shared neural network only when episode ends or when at specified frequency
                if total_step % UPDATE_GLOBAL_ITER or done:

                    # Update shared neural network by local loss of gradients from local neural network
                    # Update local neural network parameters by the updated shared neural network parameters
                    push_and_pull(
                        shared_optimizer=self.opt,
                        local_network=self.lnet,
                        shared_network=self.gnet,
                        done=done,
                        next_state=s_,
                        buffer_state=buffer_s,
                        buffer_action=buffer_a,
                        buffer_reward=buffer_r,
                        gamma=GAMMA
                    )

                    # Every time updating, clear experience
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        record(
                            global_episode=self.g_ep,
                            global_episode_reward=self.g_ep_r,
                            local_episode_reward=ep_r,
                            res_queue=self.res_queue,
                            name=self.name
                        )
                        break

                # Iterate to the next state
                s = s_
                total_step += 1

        # Append None at the end of list of total rewards to call .join() to stop parallel programming
        self.res_queue.put(None)


if __name__ == '__main__':

    # Parallel programming
    print('Start main')
    print(f'Module name: {__name__}')
    print(f'Parent process: {os.getppid()}')
    print(f'Process ID: {os.getpid()}')
    print()

    # Shared network
    global_network = Network(N_S, N_A)

    # Share the shared network parameters in multiprocessing
    global_network.share_memory()

    # Shared optimizer
    global_optimizer = SharedAdam(global_network.parameters(), lr=LR, betas=(BETA_01, BETA_02))

    # Initialize shared episode counter, shared ?, and shared ?
    global_ep = mp.Value('i', 0)  # i indicates a signed integer, mp.Value to store values in a shared memory
    global_ep_r = mp.Value('d', 0.)  # d indicates a double precision float, mp.Value to store values in a shared memory
    res_queue = mp.Queue()  # Queue allows multiple processes to access the same queue object, meaning we don't need Lock
    # print(f'global_ep: {global_ep}')
    # print(f'global_ep_r: {global_ep_r}')
    # print(f'res_queue: {res_queue}')

    # Parallel training

    # Make parallel agents by mp.Process subclass Worker
    print('Make parallel agents')
    workers = [Worker(global_network, global_optimizer, global_ep, global_ep_r, res_queue, i)
               for i in range(NUM_WORKERS)]

    # Start parallel training
    print()
    print('*** Start parallel training ***')
    [w.start() for w in workers]

    total_rewards = []
    while True:
        reward = res_queue.get()
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
    torch.save(global_network.state_dict(), MODEL)

    # Show rewards
    plt.plot(total_rewards)
    plt.xlabel('Step')
    plt.ylabel('Moving average ep reward')
    plt.show()
