import gym
import tensorflow as tf
import numpy as np


def update_target(target_weights, weights, tau):
    """
    Soft target update
    :param target_weights:
    :param weights:
    :param tau:
    :return:
    """
    for (target, non_target) in zip(target_weights, weights):
        target.assign(target * (1 - tau) + non_target * tau)
        return target


def policy(state, noise_object, actor_model, bound):
    """

    :param state:
    :param noise_object: In practice, this is Ornstein-Uhlenbeck process in DDPG
    :param actor_model:
    :return:
    """
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Add noise to actions
    sampled_actions = sampled_actions.numpy() + noise
    # Adjust continuous actions to be within the permitted bound
    legal_action = np.clip(sampled_actions, -bound, bound)
    return [np.squeeze(legal_action)]


def environment_spec(env):
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    bound = env.action_space.high[0]
    print('***** About environment ******')
    print(f'Environment name: {ENV}')
    print(f'State space: {num_states}')
    print(f'Action space: {num_actions}')
    print(f'Action bound: {bound}')
    print('******************************')
    return num_states, num_actions, bound