import gym
import numpy as np
import tensorflow as tf
from tensorflow as keras
from tensorflow.keras import layers

# parameters
seed = 42
gamma = 0.99
max_steps_per_episode = 10000
ENV = 'MountainCar-v0'
eps = np.finfo(np.float32).eps.item() # Smallest number s.t. 1.0 + eps != 1.0

# set seed
tf.random.set_seed(seed)
np.random.seed(seed)

# environment
env = gym.make(ENV)
env.seed(seed)
print('state', env.observation_space)
print('action', env.action_space)
print('reward', env.reward_range)
sample = env.reset()
print('state example', sample)
