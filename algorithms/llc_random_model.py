import gym
import numpy as np
import imageio


# Parameters
ENV = 'LunarLanderContinuous-v2'
VIDEO = '../videos/lcc_random.gif'
SEED = 0
BOUND_MAIN = 1.0
BOUND_SIDE = 1.0
EVAL_EPISODE = 3
np.random.seed(SEED)


# Environment
env = gym.make(ENV)
env.seed(SEED)


# Make gif
with imageio.get_writer(VIDEO, fps=30) as video:

    for i in range(EVAL_EPISODE):
        env.reset()
        done = False
        screen = env.render(mode='rgb_array')
        video.append_data(screen)
        t = 1

        while not done:
            action_main = np.random.uniform(low=-BOUND_MAIN, high=BOUND_MAIN, size=1)
            action_side = np.random.uniform(low=-BOUND_SIDE, high=BOUND_SIDE, size=1)
            action = np.array([action_main, action_side]).reshape((2,))
            state, _, done, _ = env.step(action)
            screen = env.render(mode='rgb_array')
            video.append_data(screen)

env.close()