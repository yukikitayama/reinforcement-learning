"""
Mountain Car Continuous
Reward:
Reward is 100 at the top of the right hill, and negative squared sum of actions
from start to goal.
"""
import gym
import numpy as np
import imageio

ENV = 'MountainCarContinuous-v0'
SEED = 0
VIDEO = '../videos/ac_mcc_random.gif'

np.random.seed(SEED)

env = gym.make(ENV)
env.seed(SEED)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
reward_range = env.reward_range
print(f'***** {ENV} spec *****')
print(f'State: {num_inputs}')
print(f'Action: {num_actions}')
print(f'Action upper: {upper_bound}')
print(f'Action lower: {lower_bound}')
print(f'Reward range: {reward_range}')
print('***********************')


# Test
print('***** Test *****')
prev_state = env.reset()
action = np.random.uniform(low=lower_bound, high=upper_bound, size=1)
print('Sampled action', action)
state, reward, done, _ = env.step(action)
print('Next state', state)
print('Reward', reward)
print('**********')

# Make gif
print('***** Make random agent *****')
sampled_states = []
sampled_actions = []
sampled_rewards = []
with imageio.get_writer(VIDEO, fps=30) as video:
    env.reset()
    done = False
    screen = env.render(mode='rgb_array')
    video.append_data(screen)
    t = 1

    while not done:
        action = np.random.uniform(low=lower_bound, high=upper_bound, size=1)
        state, reward, done, _ = env.step(action)
        screen = env.render(mode='rgb_array')
        video.append_data(screen)
        # Debug
        sampled_states.append(state)
        sampled_actions.append(action)
        sampled_rewards.append(reward)
        t += 1
        if t > 190:
            print('Break')
            break

env.close()

