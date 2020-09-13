from tensorflow import keras
from tensorflow.keras import layers


def get_actor_critic(num_inputs, num_actions, num_hidden):
    inputs = layers.Input(shape=(num_inputs,), name='input_layer')
    common = layers.Dense(num_hidden, activation='relu', name='common_dense_layer')(inputs)
    # actor
    action = layers.Dense(num_actions, activation='softmax', name='actor_output_layer')(common)
    # critic
    critic = layers.Dense(1, activation='linear', name='critic_output_layer')(common)
    model = keras.Model(inputs=inputs, outputs=[action, critic])
    return model
