from algorithm.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import gym
from gym import wrappers
import os


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


def get_make_env_fn(**kargs):

    def make_env_fn(env_name, seed=None, render=None, record=False,
                    unwrapped=False, monitor_mode=None,
                    inner_wrappers=None, outer_wrappers=None):
        mdir = tempfile.mkdtemp()
        env = None
        if render:
            try:
                env = gym.make(env_name, render=render)
            except:
                pass
        if env is None:
            env = gym.make(env_name)
        if seed is not None: env.seed(seed)
        env = env.unwrapped if unwrapped else env
        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)
        env = wrappers.Monitor(
            env, mdir, force=True,
            mode=monitor_mode,
            video_callable=lambda e_idx: record) if monitor_mode else env
        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)
        return env

    return make_env_fn, kargs


def check_make_env_fn():
    env_name = 'CartPole-v1'
    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
    print('make_env_fn', make_env_fn)
    print('make_env_kargs', make_env_kargs)
    env = make_env_fn(**make_env_kargs)
    print(env)
    state = env.reset()
    tmp = env.step(0)
    print(tmp)


def save_checkpoint_path():
    curr = os.getcwd()
    print('Current working directory\n', curr)
    print()

    checkpoint_dir = '/home/yuki/PycharmProjects/reinforcement-learning/model'
    episode_idx = 100
    path = os.path.join(checkpoint_dir, f'model.{episode_idx}.tar')
    print('Checkpoint path', path)


def main():

    # visualize_beta()

    # visualize_epsilon_decay()

    # mdir = tempfile.mkdtemp()
    # print(mdir)

    # check_make_env_fn()

    save_checkpoint_path()


if __name__ == '__main__':
    main()
