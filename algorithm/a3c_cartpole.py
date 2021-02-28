import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gym

# Parameter
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Device: {device}')
print(f'Number of CPUs: {mp.cpu_count()}')
env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


# Initialize weights in layers in neural network
def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)


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
        set_init([self.pi1, self.pi2, self.v1, self.v2])


def main():

    # Global network
    global_network = Network(N_S, N_A)

    # Global optimizer

    # ?
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()
    print(f'global_ep: {global_ep}')
    print(f'global_ep_r: {global_ep_r}')
    print(f'res_queue: {res_queue}')


if __name__ == '__main__':
    main()
