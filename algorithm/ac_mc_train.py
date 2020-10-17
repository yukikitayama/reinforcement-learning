"""
OpenAI Gym MountainCar-v0, https://github.com/openai/gym/wiki/MountainCar-v0
Solving is defined as reward of -110.0 over 100 consecutive trials. Episode ends
when we reach 0.5 position (goal position), or if 200 iterations are reached.

Model is updated after one episode finishes.
"""
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# my utils
from algorithms.ac_mc_model import get_actor_critic

# parameters
seed = 42
gamma = 0.99
max_steps_per_episode = 10000
ENV = 'MountainCar-v0'
"""
We add a small value to the denominator when we normalize returns by std to
avoid zero denominator division. 
"""
eps = np.finfo(np.float32).eps.item()  # Smallest number s.t. 1.0 + eps != 1.0
MODEL = 'actor_critic_mountain_car.h5'

# set seed
tf.random.set_seed(seed)
np.random.seed(seed)

# environment
env = gym.make(ENV)
env.seed(seed)
# [position, velociy]
print('state', env.observation_space)
# 0: push left, 1: no push, 2: push right
print('action', env.action_space)
# -1 for each time steps until goal position of 0.5 is reached
print('reward', env.reward_range)
sample = env.reset()
print('state example', sample)
# sample = env.render(mode='rgb_array')
# plt.imshow(sample)
# plt.show()

num_inputs = 2
num_actions = 3
num_hidden = 128

# model
# inputs = layers.Input(shape=(num_inputs,), name='input_layer')
# common = layers.Dense(num_hidden, activation='relu', name='common_dense_layer')(inputs)
# # actor
# action = layers.Dense(num_actions, activation='softmax', name='actor_output_layer')(common)
# # critic
# critic = layers.Dense(1, activation='linear', name='critic_output_layer')(common)
# model = keras.Model(inputs=inputs, outputs=[action, critic])


# training
model = get_actor_critic(num_inputs, num_actions, num_hidden)
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
# training will not finish until we can solve this environment
while True:
    # initialize
    state = env.reset()
    episode_reward = 0

    with tf.GradientTape() as tape:

        for timestep in range(1, max_steps_per_episode):

            state = tf.convert_to_tensor(state)  # (2,)
            state = tf.expand_dims(state, 0)  # (1, 2)

            # predict action probabilities for each action and value of this
            # state (estimated future rewards)
            action_probs, critic_value = model(state)
            # [0, 0] and [0][0] are same
            critic_value_history.append(critic_value[0, 0])

            # get action. np.squeeze converts tensor into np array
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            # store action probability of the chosen action in log format
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # get next state and reward
            state, reward, done, _ = env.step(action)

            # store reward in each timestep in each episode
            rewards_history.append(reward)

            # Accumulate rewards within one episode
            episode_reward += reward

            if done:
                # debug
                print('state at done', state[0])

                # Break for for loop
                break

        # Running reward is slowly updated by accumulated rewards from each
        # episode.
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected reward from rewards
        returns = []
        discounted_sum = 0
        """
        Reverse rewards history and insert at the beginning, so the items 
        inserted first are pushed backward in inserting, so the result is still 
        ordered correctly. Recent reward has less affected by gamma.
        """
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize by np array and make it back list
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        # Calculate loss to update network
        for log_prob, value, ret in history:

            # actor loss
            # Is diff advantage?
            diff = ret - value
            # - log probability multiplied by advantage will be policy gradient
            actor_losses.append(-log_prob * diff)

            # critic loss
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        # Get gradients
        grads = tape.gradient(loss_value, model.trainable_variables)
        # Gradient ascent
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear history of action probabilities, critic values, and rewards
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

        # Go to the next episode

    # Monitoring
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))
        model.save_weights(MODEL)
        print('Saved model')

    # Break for the most outside while loop
    if running_reward > 195:
        print("Solved at episode {}".format(episode_count))
        break

    # Test
    if episode_count == 1000:
        print('Currently only testing so break training')
        break
