import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Dense, Input
import matplotlib.pyplot as plt
import pickle
import imageio


# Parameters
ENV = 'CartPole-v0'
PATH_REWARD_01 = '../objects/dqn_cartpole_ep_reward.pkl'
PATH_REWARD_02 = '../objects/dqn_cartpole_ma_reward.pkl'
PATH_SAVEFIG_01 = '../images/dqn_cartpole_episode_reward.png'
PATH_SAVEFIG_02 = '../images/dqn_cartpole_average_reward.png'
PATH_MODEL = '../model/dqn_cartpole_target_model_1000.h5'
PATH_VIDEO = '../videos/dqn_cartpole.gif'
EPISODE = 5
FPS = 30


def get_model(num_states, num_actions):
    inputs = Input(shape=(num_states,))
    out = Dense(128, activation='relu')(inputs)
    out = Dense(128, activation='relu')(out)
    outputs = Dense(num_actions, activation='linear')(out)
    model = Model(inputs, outputs)
    return model


def main():
    # Episode reward
    ep_reward = pickle.load(open(PATH_REWARD_01, 'rb'))
    plt.plot(ep_reward)
    plt.savefig(PATH_SAVEFIG_01)
    plt.show()

    # Average reward
    avg_reward = pickle.load(open(PATH_REWARD_02, 'rb'))
    plt.plot(avg_reward)
    plt.savefig(PATH_SAVEFIG_02)
    plt.show()

    # Environment
    env = gym.make(ENV)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Model
    # model = Model(num_states=num_states, num_actions=num_actions)
    model = get_model(num_states=num_states, num_actions=num_actions)
    model.load_weights(PATH_MODEL)

    # Evaluation video
    with imageio.get_writer(PATH_VIDEO, fps=FPS) as video:

        for i in range(EPISODE):

            # Initialize
            state = env.reset()
            done = False
            screen = env.render(mode='rgb_array')
            video.append_data(screen)

            # Start episode
            while not done:

                # Get action
                action = model(np.atleast_2d(state.astype('float32')))
                action = np.argmax(action[0])

                # Get next state
                next_state, _, done, _ = env.step(action)
                screen = env.render(mode='rgb_array')
                video.append_data(screen)
                state = next_state

            print(f'Episode {i+1} finished')

    env.close()


if __name__ == '__main__':
    main()
