import gym
import tensorflow as tf
import numpy as np
# my utils
from algorithms.ac_mc_model import get_actor_critic
import imageio

MODEL = 'actor_critic_mountain_car.h5'
ENV = 'MountainCar-v0'
NUM_INPUTS = 2
NUM_ACTIONS = 3
NUM_HIDDEN = 128
VIDEO = '/home/yuki/PycharmProjects/reinforcement-learning/videos/actor_critic_mountaincar.gif'

# environment
env = gym.make(ENV)

# model
model = get_actor_critic(NUM_INPUTS, NUM_ACTIONS, NUM_HIDDEN)

# test predict
state = env.reset()
state = tf.convert_to_tensor(state)
state = tf.expand_dims(state, 0)
print('prediction', model(state))

# make gif
with imageio.get_writer(VIDEO, fps=30) as video:
    prev_state = env.reset()
    done = False
    screen = env.render(mode='rgb_array')
    video.append_data(screen)

    while not done:
        prev_state = tf.convert_to_tensor(prev_state)
        prev_state = tf.expand_dims(prev_state, axis=0)
        # Evaluation does not need critic value
        action_probs, _ = model(prev_state)
        action = np.random.choice(NUM_ACTIONS, p=np.squeeze(action_probs))
        # Reward is also unnecessary in evaluation
        state, _, done, _ = env.step(action)
        screen = env.render(mode='rgb_array')
        video.append_data(screen)
        # Go to next time step
        prev_state = state

env.close()