# Setup
import gym
import numpy as np
from tensorflow.keras.optimizers import Adam
from algorithms.llc_model import actor, critic
from algorithms import OUActionNoise
from algorithms.util import update_target, policy, environment_spec


# Hyperparameters
ENV = 'LunarLanderContinuous-v2'
STD_DEV = 0.2
LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.002
TOTAL_EPISODES = 100


# Training
def main():
    # Get environment information
    env = gym.make(ENV)
    num_states, num_actions, bound = environment_spec(env)

    # Instantiate noise object
    ou_noise = OUActionNoise(mean=np.zeros(1), std=float(STD_DEV) * np.ones(1))

    # Initialize models
    # Non-target neural network
    actor_model = actor(num_states, bound)
    critic_model = critic(num_states, num_actions)
    # Target neural network
    target_actor = actor(num_states, bound)
    target_critic = critic(num_states, num_actions)

    # Get optimizers for neural network
    actor_optimizer = Adam(LEARNING_RATE_ACTOR)
    critic_optimizer = Adam(LEARNING_RATE_CRITIC)

if __name__ == "__main__":
    main()