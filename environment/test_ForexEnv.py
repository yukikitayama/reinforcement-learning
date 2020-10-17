from environment.ForexEnv import ForexEnv
import matplotlib.pyplot as plt


# Parameter
LENGTH = 200
SEED = 0


# Test constructor
env = ForexEnv(episode_length=LENGTH)
print(env.size, env.index, env.episode_length)

# Test set_seed method
env.set_seed(seed_int=SEED)

# Test reset method
env.reset()
print(env.index)

# Test step method
state, done = env.step()
print('state', state, 'done', done)

# Test iteration
env.reset()
done = False
states = []
while not done:
    next_state, done = env.step()
    # print(next_state, done)
    states.append(next_state)
plt.plot(states)
plt.show()
