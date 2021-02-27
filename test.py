import gym
import matplotlib.pyplot as plt
import torch
import tensorflow as tf


def check_gym():
    env = gym.make('SpaceInvaders-v0')
    print('action space', env.action_space)
    print('observation space', env.observation_space)

    obs = env.reset()
    plt.imshow(obs)
    plt.show()


def check_pytorch():
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')


def check_tensorflow():
    print(tf.config.list_physical_devices('GPU'))


def main():

    # Gym
    check_gym()

    # Pytorch
    check_pytorch()

    # Tensorflow
    check_tensorflow()


if __name__ == '__main__':
    main()