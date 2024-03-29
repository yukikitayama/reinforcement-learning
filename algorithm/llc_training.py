# Setup
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pickle
# User defined classes and functions
from algorithms.llc_model import actor, critic
from algorithms.OUActionNoise import OUActionNoise
from algorithms.Buffer import Buffer
from algorithms.util import update_target, policy, environment_spec, \
    monitoring


# Hyperparameters
ENV = 'LunarLanderContinuous-v2'
STD_DEV = 0.2
LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.002
EPISODES_TRAINING = 1000
GAMMA = 0.99
TAU = 0.005
BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
MONITOR_INTERVAL = 1
SAVE_INTERVAL = 10
MA = 10  # to calculate moving average of rewards
SEED = 0
# True if you want to visualize agent and environment of each training, and if
# it's not using Jupyter Notebook, because env.render() does not work in
# Notebook.
RENDER = True
PATH_REWARD_01 = '../objects/llc_ddpg_ma_reward.pkl'
PATH_REWARD_02 = '../objects/llc_ddpg_ep_reward.pkl'
PATH_MODEL_01 = '../model/llc_ddpg_target_actor.h5'
PATH_MODEL_02 = '../model/llc_ddpg_target_critic.h5'


# Training
def main():
    # Get environment information
    env = gym.make(ENV)
    num_states, num_actions, bound = environment_spec(env)

    # Set seed
    env.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Instantiate noise object
    ou_noise = OUActionNoise(mean=np.zeros(1), std=float(STD_DEV) * np.ones(1))

    # Initialize models
    # Non-target neural network
    actor_model = actor(num_states, num_actions, bound)
    critic_model = critic(num_states, num_actions)
    # Target neural network
    target_actor = actor(num_states, num_actions, bound)
    target_critic = critic(num_states, num_actions)

    # Get optimizers for neural network
    actor_optimizer = Adam(LEARNING_RATE_ACTOR)
    critic_optimizer = Adam(LEARNING_RATE_CRITIC)

    # Get experience replay buffer
    buffer = Buffer(buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE,
                    num_states=num_states, num_actions=num_actions,
                    actor_model=actor_model, critic_model=critic_model,
                    target_actor=target_actor, target_critic=target_critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    gamma=GAMMA)

    # Store rewards of each episodes
    ep_reward = []

    # Store average rewards of the last several episodes
    avg_reward = []

    for ep in range(EPISODES_TRAINING):

        # Initialize training
        prev_state = env.reset()
        episode_reward = 0

        while True:

            # Visualize agent and environment in each training episode
            if RENDER:
                env.render()

            # Convert array to tensor
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            # Get continuous action
            action = policy(state=tf_prev_state, noise_object=ou_noise,
                            actor_model=actor_model, bound=bound)

            # Get reward and next state
            state, reward, done, info = env.step(action)
            # Collect experience
            buffer.record((prev_state, action, reward, state))
            episode_reward += reward

            # Update non-target actor and critic neural network
            buffer.learn()
            # Update target actor and critic neural network
            update_target(target_actor.variables, actor_model.variables,
                          tau=TAU)
            update_target(target_critic.variables, critic_model.variables,
                          tau=TAU)

            # End a episode by breaking while loop when done variable is True
            if done:
                break

            # Iterate to the next state
            prev_state = state

        # Collect the total reward of each episode
        ep_reward.append(episode_reward)

        # Get moving average reward to observe if there is improvement
        ma_reward = np.mean(ep_reward[-MA:])

        # Monitor training
        if ep % MONITOR_INTERVAL == 0:
            monitoring(curr_episode=ep, ma_reward=ma_reward)

        # Collect the moving average reward for evaluation
        avg_reward.append(ma_reward)

        # Periodically save training result
        if ep % SAVE_INTERVAL == 0:
            # Moving average reward
            pickle.dump(avg_reward, open(PATH_REWARD_01, 'wb'))
            # Total rewards of each episode
            pickle.dump(ep_reward, open(PATH_REWARD_02, 'wb'))
            # Target actor
            target_actor.save_weights(PATH_MODEL_01)
            # Target critic
            target_critic.save_weights(PATH_MODEL_02)
            print('************************************')
            print('Saved rewards and models temporarily')
            print('************************************')

    # Save training results
    # Moving average of rewards
    pickle.dump(avg_reward, open(PATH_REWARD_01, 'wb'))
    # Total rewards of each episode
    pickle.dump(ep_reward, open(PATH_REWARD_02, 'wb'))
    # Target actor
    target_actor.save_weights(PATH_MODEL_01)
    # Target critic
    target_critic.save_weights(PATH_MODEL_02)
    print('***************************************')
    print('Saved rewards and models after training')
    print('***************************************')


if __name__ == "__main__":
    main()
