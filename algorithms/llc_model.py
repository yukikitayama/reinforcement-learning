"""
Lunar lander continuous

State is 8 dimensions, and action is 2 continuous dimensions.
"""


# Setup
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, \
    Concatenate


# Parameters
TANH_INIT = 0.003


# Models
def actor(num_states, num_actions, bound):
    inputs = Input(shape=(num_states,), name='input')
    hidden = Dense(units=256, activation='relu',
                   kernel_initializer='he_uniform', name='dense_1')(inputs)
    hidden = BatchNormalization(name='batch_norm_1')(hidden)
    hidden = Dense(units=256, activation='relu',
                   kernel_initializer='he_uniform', name='dense_2')(hidden)
    hidden = BatchNormalization(name='batch_norm_2')(hidden)
    # Initialize weights in output layer between -3e-3 and 3-e3
    outputs_init = tf.random_uniform_initializer(minval=-TANH_INIT,
                                                 maxval=TANH_INIT)
    outputs = Dense(units=num_actions, activation='tanh',
                    kernel_initializer=outputs_init, name='output')(hidden)
    # outputs = outputs * bound
    outputs = tf.multiply(outputs, bound)
    model = Model(inputs=inputs, outputs=outputs, name='actor')
    return model


def critic(num_states, num_actions):
    """
    This Critic outpus Q(s,a) so it requires both state and action inputs
    :param num_states:
    :return:
    """
    # State input
    state_inputs = Input(shape=(num_states), name='state_input')
    state_hidden = Dense(units=16, activation='relu',
                         kernel_initializer='he_uniform',
                         name='state_dense_1')(state_inputs)
    state_hidden = BatchNormalization(name='state_batch_norm_1')(state_hidden)
    state_hidden = Dense(units=32, activation='relu',
                         kernel_initializer='he_uniform',
                         name='state_dense_2')(state_hidden)
    # Action input
    action_inputs = Input(shape=(num_actions))
    action_hidden = Dense(units=32, activation='relu',
                          kernel_initializer='he_uniform',
                          name='action_dense_1')(action_inputs)
    action_hidden = BatchNormalization(name='action_batch_norm_1')(action_hidden)
    # Concatenate state and action layers
    common = Concatenate(name='concat_1')([state_hidden, action_hidden])
    common = Dense(units=256, activation='relu',
                   kernel_initializer='he_uniform',
                   name='common_dense_1')(common)
    common = BatchNormalization(name='common_batch_norm_1')(common)
    common = Dense(units=256, activation='relu',
                   kernel_initializer='he_uniform',
                   name='common_dense_2')(common)
    common = BatchNormalization(name='common_batch_norm_2')(common)
    outputs = Dense(units=1, activation='linear',
                    name='output')(common)
    model = Model(inputs=[state_inputs, action_inputs],
                  outputs=outputs, name='critic')
    return model
