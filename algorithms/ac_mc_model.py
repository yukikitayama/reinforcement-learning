import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


def get_actor_critic(num_inputs, num_actions, num_hidden):
    # Input
    inputs = Input(shape=(num_inputs,), name='input_layer')
    # Shared layer
    common = Dense(units=num_hidden, activation='relu', name='common_dense_layer')(inputs)
    # Actor
    action = Dense(units=num_actions, activation='softmax', name='actor_output_layer')(common)
    # Critic
    critic = Dense(units=1, activation='linear', name='critic_output_layer')(common)
    # Combine
    model = Model(inputs=inputs, outputs=[action, critic])
    return model


def actor_critic_continuous(num_inputs, num_hidden, bound):
    """
    Initialize weights of Actor last layer to be between -0.003 and 0.003 as
    this avoid getting -1 or 1 output in the initial stage, which makes gradient
    zero.
    :param num_inputs:
    :param num_hidden:
    :param bound:
    :return:
    """
    # Input
    inputs = Input(shape=(num_inputs,), name='input_layer')
    # Shared layer
    common = Dense(units=num_hidden, activation='relu',
                   kernel_initializer='he_uniform', name='common_layer_1')(inputs)
    # common = Dense(units=num_hidden, activation='relu', name='common_layer_2')(common)
    # Actor
    init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    action = Dense(units=1, activation='tanh',
                   kernel_initializer=init, name='actor_output_layer')(common)
    action = action * bound
    # Critic
    critic = Dense(units=1, activation='linear',
                   kernel_initializer='glorot_uniform', name='critic_output_layer')(common)
    # Combine
    model = Model(inputs=inputs, outputs=[action, critic])
    return model