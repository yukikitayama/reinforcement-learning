"""
Experience is sampled with probabilities which are made by TD error priorities. But when you calculate the loss to
update network, the difference between the target and the estimate is multiplied by the weights which are calculayed by
the probabilities. With the higher priorities, the experience is sampled more, but the effect on network update is
lessened. Check the function of per_example.
"""
import numpy as np
import gym
import torch


class PrioritizedReplayBuffer:
    def __init__(self, max_samples=10000, batch_size=64, rank_based=False,
                 alpha=0.6, beta0=0.1, beta_rate=0.99992, eps=1e-6):
        self.max_samples = max_samples
        # Make a new array of given shape and data type without initialization
        # 2? ndarray?
        self.memory = np.empty(shape=(self.max_samples, 2),
                               dtype=np.ndarray)
        self.batch_size = batch_size
        # Current size of the replay buffer
        self.n_entries = 0
        # ?
        self.next_index = 0
        # ?
        self.td_error_index = 0
        # ?
        self.sample_index = 1
        self.rank_based = rank_based  # If not rank_based, n proportional
        self.alpha = alpha
        # beta is current beta while beta0 is the initial beta
        self.beta = beta0
        self.beta0 = beta0  # beta0 is beta's initial value
        self.beta_rate = beta_rate
        # eps is a small value added to denominator to calculate priority from TD error
        self.eps = eps

    def update(self, idxs, td_errors):
        """
        Update what?
        """
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)

        if self.rank_based:
            # np.argsort returns indices that sorts an array
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def store(self, sample):
        """
        What is sample?
        """
        priority = 1.0

        if self.n_entries > 0:
            # Why getting the highest priority up to n_entries?
            priority = self.memory[:self.n_entries, self.td_error_index].max()

        # Why putting the calculated priority to the next index location?
        self.memory[self.next_index, self.td_error_index] = priority
        self.memory[self.next_index, self.sample_index] = np.array(sample)

        # Update the index for replay buffer
        self.n_entries = min(self.n_entries + 1, self.max_samples)

        # next_index goes back to the start when exceeding the maximum size
        self.next_index += 1
        self.next_index = self.next_index % self.max_samples

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        self._update_beta()

        # memory is initialized with empty values, so get data up to n_entries if the memory is not full
        entries = self.memory[:self.n_entries]
        # Rank-based prioritization
        if self.rank_based:
            priorities = 1 / (np.arange(self.n_entries) + 1)
        # Proportional prioritization
        else:
            priorities = entries[:, self.td_error_index] + self.eps

        scaled_priorities = priorities ** self.alpha
        # Priority real values to probability
        probs = np.array(scaled_priorities / np.sum(scaled_priorities), dtype=np.float64)
        # Weighted importance-sampling weights
        weights = (self.n_entries * probs) ** -self.beta
        # Downscale the weights so that the largest weights are 1, and everything else is lower
        normalized_weights = weights / weights.max()

        idxs = np.random.choice(self.n_entries, size=batch_size, replace=False)
        idxs_stack = np.vstack(idxs)

        # Each sample is 2 dims, td error index and sample index
        samples = np.array([entries[idx] for idx in idxs])
        # ?
        samples_stack = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]

        weights_stack = np.vstack(normalized_weights[idxs])

        return idxs_stack, weights_stack, samples_stack

    def _update_beta(self):
        """
        beta is between 0 and 1. We start with small beta and get 1 towards the end of training. beta_rate ** -1 is
        a real number more than 1.0. So every time _update_beta, beta gets bigger slowly towards 1.
        """
        self.beta = min(1.0, self.beta * self.beta_rate ** -1)
        return self.beta

    def __len__(self):
        return self.n_entries

    def __repr__(self):
        return str(self.memory[:self.n_entries])

    def __str__(self):
        return 'Prioritized Experience Replay\n' + str(self.memory[:self.n_entries])


class PrioritizedReplay:
    """

    https://github.com/ShangtongZhang/DeepRL/blob/932ea88082e0194126b87742bd4a28c4599aa1b8/deep_rl/component/replay.py#L152
    """
    def __init__(self, memory_size):
        self.tree = SumTree(memory_size)
        self.max_priority = 1

    def feed(self, data):
        return None

    def sample(self, batch_size):
        # ?
        segment = self.tree.total() / batch_size

        sampled_data = []

    def update_priorities(self, info):
        for idx, priority in info:
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)


class SumTree:
    """
    A binary tree data structure. A parent value is the sum of all the subsequent children's values.
    An example of SumTree
         13
        /  \
      10    3
     /  \    \
    4    6    3
    This binary tree is implemented by array representation
    idx    0  1  2  3  4  5
    value 13 10  3  4  6  3
    If a parent is p, left child is 2 * p + 1, and right child is 2 * p + 2

    _retrieve methods recursively gose to the bottom of the tree and return the index

    https://www.fcodelabs.com/2019/03/18/Sum-Tree-Introduction/
    https://github.com/ShangtongZhang/DeepRL/blob/932ea88082e0194126b87742bd4a28c4599aa1b8/deep_rl/utils/sum_tree.py#L6
    https://adventuresinmachinelearning.com/sumtree-introduction-python/

    leaf nodes correspond to the weights.

    """

    # Class variable, shared across all the objects of this class, modifying a class variable affects all objects
    # instance at the same time.
    write = 0

    def __init__(self, capacity):
        # Instance variable
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        What is s?
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        # Decide left or right you want to go down further
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """
        Root (self.tree[0]) contains the total sum of all the children for this SumTree data structure.
        """
        return self.tree[0]

    def add(self, p, data):
        """
        Store priority and sample. Probably argument data is sample
        """
        # ?
        idx = self.write + self.capacity - 1
        # pending_idx is a set
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """
        Update priority
        idx is an index for a value in tree to be updated, and p is priority
        """
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """
        Get priority and sample
        What is s?
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.tree[idx], dataIdx)


def per_example():
    # probs are probabilities from priorities
    probs = np.array([0.2, 0.8])
    print('Probability', probs)
    n = 100

    beta = 0.1
    print('beta', beta)
    w = (n * probs) ** (-1 * beta)
    print('w', w)
    w_norm = w / max(w)
    print('w_norm', w_norm)

    beta = 0.5
    print('beta', beta)
    w = (n * probs) ** (-1 * beta)
    print('w', w)
    w_norm = w / max(w)
    print('w_norm', w_norm)

    beta = 1.0
    print('beta', beta)
    w = (n * probs) ** (-1 * beta)
    print('w', w)
    w_norm = w / max(w)
    print('w_norm', w_norm)


def test_beta_update():
    beta = 0.1
    beta_rate = 0.9
    print('beta:', beta)
    print('beta_rate ** -1:', beta_rate ** -1)
    beta = beta * beta_rate ** -1
    print('1:', beta)
    beta = beta * beta_rate ** -1
    print('2:', beta)
    beta = beta * beta_rate ** -1
    print('3:', beta)


def main():

    # test_beta_update()

    env = gym.make('CartPole-v0')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    state1 = env.reset()

    state, reward, done, info = env.step(0)

    # Loss is the difference between target q value and estimated q value
    loss1 = 1.1 - 1.0
    loss2 = 1.5 - 1.0
    replay_eps = 0.01
    replay_alpha = 0.5
    replay_beta = 0.4
    idx = [1, 2]
    loss = torch.tensor(data=[loss1, loss2], dtype=torch.float, device=device)
    print()
    print('loss computed from target q and estimated q\n', loss)
    print()
    # .add(x) is element-wisely add x
    # p_i = |delta_i| + epsilon -> p_i^alpha
    priorities = loss.abs().add(replay_eps).pow(replay_alpha)
    print('priorities computed from the computed loss\n', priorities)
    print()
    idxs = torch.from_numpy(np.asarray(idx, dtype=np.float32)).to(device).long()
    print('idxs sampled from prioritized replay\n', idxs)
    print()

    # update_priorities
    print('[ Update priorities by the sampled index and the computed priorities ]')
    print()
    idxs_np = idxs.cpu().detach().numpy()
    priorities_np = priorities.cpu().detach().numpy()
    # print('idxs_np', idxs_np)
    # print('priorities_np', priorities_np)

    sampling_prob = torch.tensor(data=[0.8, 0.2], dtype=torch.float, device=device)
    print('sampling_prob sampled from prioritized replay\n', sampling_prob)
    print()
    # print('sampling_prob.size(0)', sampling_prob.size(0))
    weights = sampling_prob.mul(sampling_prob.size(0)).add(1e-6).pow(-replay_beta)
    print('weights computed from sampling_prob from prioritized replay\n', weights)
    print()
    weights = weights / weights.max()
    print('weights after normalizing by max\n', weights)
    print()
    loss = loss.mul(weights)
    print('loss multiplied by the weights\n', loss)

    # How can I make idx and sampling_prob from transitions

    # PrioritizedTransition is (state, action, reward, next_state, mask, sampling_prob, idx)


if __name__ == '__main__':
    main()
