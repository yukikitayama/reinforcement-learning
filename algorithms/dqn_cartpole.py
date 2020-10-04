import gym
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

# Parameters
ENV = 'CartPole-v0'


class DQN(tf.keras.Model):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        self.input = InputLayer(input_shape=(num_states,))
        self.hidden_01 = Dense(units=256, activation='relu')
        self.hidden_02 = Dense(units=256, activation='relu')
        self.output = Dense(units=num_actions, activation='linear')

    @tf.function
    def call(self, inputs):
        x = self.input(inputs)
        x = self.hidden_01(x)
        x = self.hidden_02(x)
        output = self.output(x)
        return output


class Agent:
    def __init__(self, num_states, num_actions, gamma, max_ep, min_ep,
                 batch_size, learning_rate):
        self.num_actions = num_actions
        self.gamma = gamma
        self.max_ep = max_ep
        self.min_ep = min_ep
        self.batch_size = batch_size
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model = DQN(num_states, num_actions)
        self.buffer = {'state': [], 'action': [], 'reward': [],
                       'next_state': [], 'done': []}


def main():



if __name__ == "__main__":
    main()