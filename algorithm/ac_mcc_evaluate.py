import gym
import tensorflow as tf
import numpy as np
import imageio
from algorithms.ac_mc_model import actor_critic_continuous
import imageio


# Parameter
ENV = 'MountainCarContinuous-v0'
MODEL = 'actor_critic_mountain_car.cont.h5'
NUM_INPUTS = 2
NUM_ACTIONS = 1
NUM_HIDDEN = 128
VIDEO = '../videos/ac_mcc.gif'
BOUND = 1.0


# Environment
env = gym.make(ENV)


# Model
model = actor_critic_continuous(NUM_INPUTS, NUM_HIDDEN, BOUND)


# Make gif
with imageio.get_writer(VIDEO, fps=30) as video:
    prev_state = env.reset()
    done = False
    screen = env.render(mode='rgb_array')
    video.append_data(screen)
    t = 0

    while not done:
        prev_state = tf.convert_to_tensor(prev_state)  # (2,)
        prev_state = tf.expand_dims(prev_state, 0)  # (1, 2)
        action, _ = model(prev_state)
        action = np.clip(action, -BOUND, BOUND)
        state, _, done, _ = env.step(action)
        screen = env.render(mode='rgb_array')
        video.append_data(screen)
        prev_state = state
        t += 1

        if t > 190:
            break

env.close()