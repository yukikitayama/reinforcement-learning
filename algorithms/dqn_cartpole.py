import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam


# Hyperparameters
ENV = 'CartPole-v0'
GAMMA = 0.99
COPY_INTERVAL = 50
EP_MAX = 10000
EP_MIN = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPISODE = 10000
EPSILON_START = 0.99
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1


class Model(tf.keras.Model):
    """
    This is a neural network that we use in Deep Q Network, where the network
    is the action-value function to output action-value, Q(s,a), the expected
    total reward. So the output layer outputs values by a linear function for
    each action.
    """
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


class ExperienceReplayBuffer:
    def __init__(self, max_size, min_size):
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = {'state': [], 'action': [], 'reward': [],
                       'next_state': [], 'done': []}

    def record(self, experience_dict):
        # Discard the oldest experience if the buffer is full
        if len(self.buffer['state']) >= self.max_size:
            for key in self.buffer.keys():
                self.buffer[key].pop(0)
        # Add new experience at the end
        for key, value in experience_dict.items():
            self.buffer[key].append(value)


class DQN:
    def __init__(self, num_states, num_actions, model, buffer, gamma, batch_size, learning_rate):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model = model
        self.target_model = model
        self.buffer = buffer

    def train_model(self, target_model):
        # Get mini batch
        index = np.random.randint(low=0, high=len(self.buffer.buffer['state']),
                                  size=self.batch_size)
        states = np.asarray([self.buffer.buffer['state'][i] for i in index])
        actions = np.asarray([self.buffer.buffer['action'][i] for i in index])
        rewards = np.asarray([self.buffer.buffer['reward'][i] for i in index])
        next_states = np.asarray([self.buffer.buffer['next_state'][i] for i in index])
        # Update neural network weights
        with tf.GradientTape() as tape:



# Helper functions
def get_action(state, epsilon, num_actions, model):
    # Exploration by epsilon
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    # Exploitation by greedy
    else:
        return np.argmax(model(state))


def update_target(non_target_model, target_model):
    weights_target = target_model.trainable_variables
    weights_non_target = non_target_model.trainable_variables
    for w_target, w_non in zip(weights_target, weights_non_target):
        w_target.assign(w_non.numpy())



def main():
    # Environment
    env = gym.make(ENV)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print(f'Environment: {ENV}\nState: {num_states}\nAction: {num_actions}')

    # Model
    model = Model()
    target_model = Model()

    # Experience replay buffer
    buffer = ExperienceReplayBuffer()

    # Deep Q Network
    agent = DQN()

    # Training
    # Initialize for training
    epsilon = EPSILON_START
    ep_reward = []

    for ep in range(EPISODE):
        # Initialize for each episode
        # Epsilon decay
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        total_reward = 0
        done = False
        state = env.reset()

        while not done:
            # Get action
            action = get_action(state, epsilon)
            # Get reward and next state
            next_state, reward, done, _ = env.step(action)
            # Accumulate reward
            total_reward += reward

            if done:
                break


if __name__ == "__main__":
    main()
