import gym
import matplotlib.pyplot as plt

env = gym.make('SpaceInvaders-v0')
print('action space', env.action_space)
print('observation space', env.observation_space)

obs = env.reset()
plt.imshow(obs)
plt.show()