import gym
import tensorflow as tf
import numpy as np

# parameter
ENV = 'CartPole-v0'
SEED = 0

# environment
env = gym.make(ENV)

# set seed
env.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)