"""
Solved requirements of Mountain Car Continuous is to get a reward over 90.
Reward is 100 when the car reaches the top of the hill ar the right.
"""


import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np
from algorithms.ac_mc_model import actor_critic_continuous


# Parameters
SEED = 0
GAMMA = 0.99
ENV = 'MountainCarContinuous-v0'
EPS = np.finfo(np.float32).eps.item()  # Avoid zero division
MODEL = 'actor_critic_mountain_car_cont.h5'
NUM_HIDDEN = 128
ADAM_LR = 0.01
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPS = 1e-7
HUBER_DELTA = 1.0
MAX_STEP = 180


# Set seed
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Environment
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


# Training
# Initialization for entire training
model = actor_critic_continuous(num_inputs, NUM_HIDDEN, lower_bound)
optimizer = Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1,
                 beta_2=ADAM_BETA_2, epsilon=ADAM_EPS)
loss = Huber(delta=HUBER_DELTA)
action_prob_history = []
critic_value_history = []
reward_history = []
running_reward = 0
episode_count = 0

while True:
    # Initialization for each episode
    state = env.reset()
    episode_reward = 0

    with tf.GradientTape() as tape:

        for t in range(1, MAX_STEP):
            state = tf.convert_to_tensor(state)  # (2,)
            state = tf.expand_dims(state, 0)  # (1, 2)
            # Get model outputs
            action, critic_value = model(state)
            # Actor
            log_action = tf.math.log(action)
            # print('before squeeze', action.shape)
            # action = np.squeeze(action)
            # print('after squeeze', action.shape)
            action = np.clip(action, lower_bound, upper_bound)
            action_prob_history.append(log_action)
            # Critic
            critic_value_history.append(critic_value[0, 0])
            # Get next state and reward
            state, reward, done, _ = env.step(action)
            # Store reward in each time stepm in each episode
            reward_history.append(reward)
            # Accumulate reward within one episode
            episode_reward += reward

            if done:
                # Debug
                print('t in done', t)
                # Break of for loop
                break

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected reward (returns) from collected rewards
        returns = []
        discounted_sum = 0
        for r in reward_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            returns.insert(0, discounted_sum)

        # Normaliza expected reward
        returns = np.array(returns)
        # + EPS avoids zero division
        returns = (returns - np.mean(returns)) / (np.std(returns) + EPS)
        returns = returns.tolist()

        # Calculate loss to update network
        history = zip(action_prob_history,
                      critic_value_history,
                      returns)
        actor_losses = []
        critic_losses = []
        for log_action, value, ret in history:

            # Actor
            diff = ret - value
            actor_losses.append(-log_action * diff)

            # Critic
            critic_losses.append(
                loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        # Get gradients
        grads = tape.gradient(loss_value, model.trainable_variables)
        # Gradient ascent
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear history for next episode
        action_prob_history.clear()
        critic_value_history.clear()
        reward_history.clear()

        # Go to next episode

    # Monitoing
    episode_count += 1
    if episode_count % 10 == 0:
        msg = 'Episode reward: {:.2f} at episode {}'
        print(msg.format(episode_reward, episode_count))
        model.save_weights(MODEL)
        print('Saved model')

    # Break of while loop
    if episode_reward > 90:
        print('Solved at episode {}'.format(episode_count))

    # Test
    if episode_count == 50:
        print('Testing: break training at episode {}'.format(episode_count))
        break
