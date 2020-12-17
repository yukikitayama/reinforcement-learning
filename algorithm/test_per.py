from algorithm.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import matplotlib.pyplot as plt
import numpy as np
import tempfile


def visualize_beta():
    b = PrioritizedReplayBuffer()
    plt.plot([b._update_beta() for _ in range(100000)])
    plt.title('PER Beta')
    plt.xlabel('Time step')
    plt.ylabel('Beta parameter of weighted importance-sampling')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()


def visualize_epsilon_decay(init_epsilon=0.7, min_epsilon=0.05, decay_steps=20000):
    epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01

    plt.subplot(1, 2, 1)
    plt.plot(epsilons)
    plt.title('Before')
    plt.ylim(0, 1)
    plt.grid()

    epsilons = epsilons * (init_epsilon - min_epsilon) + min_epsilon

    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title('After')
    plt.ylim(0, 1)
    plt.grid()

    plt.tight_layout()
    plt.show()


def main():

    # visualize_beta()

    # visualize_epsilon_decay()

    # mdir = tempfile.mkdtemp()
    # print(mdir)


if __name__ == '__main__':
    main()
