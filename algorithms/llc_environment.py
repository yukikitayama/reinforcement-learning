"""
LunarLanderContinuous
https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

State space is 8 dimensions
0: horizontal coordinate
1: vertical coordinate
2: horizontal speed
3: vertical speed
4: angle
5: angular speed
6: 0 or 1, 1 if first leg has contact to moon else 0
7: 0 or 1, 1 if second leg has contact else 0

Action space
"""

import gym
import matplotlib.pyplot as plt

# Parameter
ENV = 'LunarLanderContinuous-v2'


# Environment
env = gym.make(ENV)
print(env.observation_space)
# Action is two floats [main engine, left-right engines]
# Main engine: -1..0 off 0..+1, when power is between 50% and 100%, o.w. off
# Left-right: -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine
#             -0.5..0.5 off
print(env.action_space)
num_states = env.observation_space.shape[0]


screen = env.render(mode='rgb_array')
plt.imshow(screen)
plt.show()


