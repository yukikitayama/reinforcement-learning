import numpy as np
import tensorflow as tf


class Buffer:
    def __init__(self, buffer_capacity, batch_size, num_states, num_actions,
                 actor_model, critic_model, target_actor, target_critic,
                 gamma, actor_optimizer, critic_optimizer):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.gamma = gamma
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def learn(self):
        """
        This performs forward propagation, calculate loss, and backward
        propagation. record_range is necessary when the buffer size is less than
        capacity.

        Actor:
        Loss function is mean of critic value Q(s,a)

        Critic:
        Loss function is mean squared of target and Q(s,a).

        :return:
        """
        # Get data to update Actor and Critic
        record_range = min(self.buffer_counter, self.buffer_capacity)
        indices = np.random.choice(record_range, self.batch_size)
        state_batch = tf.convert_to_tensor(self.state_buffer[indices])
        action_batch = tf.covert_to_tensor(self.action_buffer[indices])
        reward_batch = tf.convert_to_tendor(self.reward_buffer[indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[indices])

        # Update Critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = (reward_batch + self.gamma
                 * self.target_critic([next_state_batch, target_actions]))
            critic_value = self.critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        # Calculate gradients
        critic_grad = tape.gradient(
            critic_loss, self.critic_model.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )
        # Updae Actor
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch)
            critic_value = self.critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
        # Calculate gradients
        actor_grad = tape.gradient(
            actor_loss, self.actor_model.trainable_variables
        )
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )