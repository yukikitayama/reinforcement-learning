import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import pickle
import time


# Hyperparameters
ENV = 'CartPole-v0'
GAMMA = 0.99
COPY_INTERVAL = 25
MONITOR_INTERVAL = 100
EPISODE = 10000
SAVE_INTERVAL = 1000
EP_MAX = 10000
EP_MIN = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPSILON_START = 0.99
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
MA_REWARD = 100
PATH_REWARD_01 = '../objects/dqn_cartpole_ep_reward.pkl'
PATH_REWARD_02 = '../objects/dqn_cartpole_ma_reward.pkl'
PATH_MODEL_01 = '../model/dqn_cartpole_target_model_final.h5'
PATH_MODEL_02 = '../model/dqn_cartpole_target_model'
PATH_EPSILON = '../objects/dqn_cartpole_epsilon.pkl'
RENDER = True


def get_model(num_states, num_actions):
    inputs = Input(shape=(num_states,))
    out = Dense(128, activation='relu')(inputs)
    out = Dense(128, activation='relu')(out)
    outputs = Dense(num_actions, activation='linear')(out)
    model = Model(inputs, outputs)
    return model


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = {'state': [], 'action': [], 'reward': [],
                       'next_state': [], 'done': []}

    def store(self, experience_dict):
        # Discard the oldest experience if the buffer is full
        if len(self.buffer['state']) >= self.max_size:
            for key in self.buffer.keys():
                self.buffer[key].pop(0)
        # Add new experience at the end
        for key, value in experience_dict.items():
            self.buffer[key].append(value)

    def size(self):
        return len(self.buffer['state'])


# Helper functions
def get_action(state, num_actions, model, epsilon):
    # Exploration by epsilon
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    # Exploitation by greedy
    else:
        return np.argmax(model.predict(np.atleast_2d(state))[0])


def update_target(non_target_model, target_model):
    depth = len(target_model.trainable_variables)
    for i in range(depth):
        weights_non_target = non_target_model.trainable_variables[i]
        target_model.trainable_variables[i].assign(weights_non_target.numpy())
    return target_model


def update_model(model, target_model, memory, min_experience, batch_size,
                 gamma, num_actions, optimizer):
    if memory.size() < min_experience:
        return 0

    # Get mini batch
    index = np.random.randint(low=0, high=len(memory.buffer['state']),
                              size=batch_size)
    states = np.asarray([memory.buffer['state'][i] for i in index])
    actions = np.asarray([memory.buffer['action'][i] for i in index])
    rewards = np.asarray([memory.buffer['reward'][i] for i in index])
    next_states = np.asarray([memory.buffer['next_state'][i] for i in index])
    dones = np.asarray([memory.buffer['done'][i] for i in index])

    next_action_values = np.max(target_model.predict(next_states), axis=1)
    # np.where allows us to have if the first argument is true, choose the
    # second argument, otherwise choose the third argument
    # done = True means it's a terminal state, so we only have a reward, and
    # no discounted action values from the next state.
    target_values = np.where(dones,
                             rewards,
                             rewards + gamma * next_action_values)

    # Update neural network weights
    with tf.GradientTape() as tape:
        action_values = tf.math.reduce_sum(
            model(np.atleast_2d(states.astype('float32'))) * tf.one_hot(actions, num_actions),
            axis=1
        )
        # Q network is trained byb minimising loss function
        loss = tf.math.reduce_mean(tf.square(target_values - action_values))
    # Gradient descent by differentiating loss function w.r.t. weights
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    # Update weights
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


def main():
    # Environment
    env = gym.make(ENV)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print(f'Environment: {ENV}\nState: {num_states}\nAction: {num_actions}')

    # Seed
    env.seed(1)
    np.random.random(1)
    tf.random.set_seed(1)

    # Model
    model = get_model(num_states=num_states, num_actions=num_actions)
    target_model = get_model(num_states=num_states, num_actions=num_actions)

    # Optimizer
    optimizer = Adam(learning_rate=LEARNING_RATE)

    # Experience replay buffer
    memory = ReplayMemory(max_size=EP_MAX)

    # Training
    # Initialize for training
    epsilon = EPSILON_START
    ep_reward = []
    avg_reward = []
    start_time = time.time()
    epsilons = []

    for ep in range(EPISODE):
        # Initialize for each episode
        total_reward = 0
        done = False
        state = env.reset()
        t = 0

        while not done:
            if RENDER:
                env.render()

            # Get action
            action = get_action(state, num_actions, model, epsilon)
            # Get reward and next state
            next_state, reward, done, _ = env.step(action)
            # Accumulate reward
            total_reward += reward

            # if done:
                # break
                # reward = -200
                # env.reset()

            # Store agent's experience
            memory.store({'state': state, 'action': action, 'reward': reward,
                          'next_state': next_state, 'done': done})

            # Train model
            update_model(model=model, target_model=target_model, memory=memory,
                         min_experience=EP_MIN, batch_size=BATCH_SIZE,
                         gamma=GAMMA, num_actions=num_actions,
                         optimizer=optimizer)

            # Count time step in an episode
            t += 1

            # Next state becomes current state
            state = next_state

            # Slowly update target model
            if t % COPY_INTERVAL == 0:
                target_model = update_target(model, target_model)

        # Epsilon decay
        epsilons.append(epsilon)
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

        # Store rewards
        ep_reward.append(total_reward)
        ma_reward = np.mean(ep_reward[-MA_REWARD:])
        avg_reward.append(ma_reward)

        # Monitor training
        if ep % MONITOR_INTERVAL == 0:
            print(f'Episode: {ep:,.0f}, ma reward: {ma_reward:,.1f}, '
                  f'ep reward: {total_reward:,.1f}, epsilon: {epsilon:,.3f}, '
                  f'minute: {((time.time() - start_time) / 60):,.1f}')

        # Periodically save results
        if ep % SAVE_INTERVAL == 0:
            path = PATH_MODEL_02 + '_' + str(ep) + '.h5'
            target_model.save_weights(path)
            pickle.dump(ep_reward, open(PATH_REWARD_01, 'wb'))
            pickle.dump(avg_reward, open(PATH_REWARD_02, 'wb'))
            pickle.dump(epsilons, open(PATH_EPSILON, 'wb'))
            print('Saved intermediate results')

    # Save final results
    pickle.dump(ep_reward, open(PATH_REWARD_01, 'wb'))
    pickle.dump(avg_reward, open(PATH_REWARD_02, 'wb'))
    target_model.save_weights(PATH_MODEL_01)
    pickle.dump(epsilons, open(PATH_EPSILON, 'wb'))
    print('Saved final results')


if __name__ == "__main__":
    main()
