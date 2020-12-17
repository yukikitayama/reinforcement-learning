"""
Experience is sampled with probabilities which are made by TD error priorities. But when you calculate the loss to
update network, the difference between the target and the estimate is multiplied by the weights which are calculayed by
the probabilities. With the higher priorities, the experience is sampled more, but the effect on network update is
lessened. Check the function of per_example.
"""
import numpy as np


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
        return str(self.memory[:self.n_entries])


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


# test_beta_update()
