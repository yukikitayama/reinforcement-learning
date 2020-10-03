# Setup
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pickle
# User defined
from algorithms.llc_model import actor
from algorithms.util import environment_spec


# Hyperparameters
ENV = 'LunarLanderContinuous-v2'
PATH_ACTOR = '../model/llc_ddpg_target_actor.h5'
PATH_VIDEO = '../videos/llc_ddpg.gif'
PATH_REWARD_01 = '../objects/llc_ddpg_ma_reward.pkl'
PATH_REWARD_02 = '../objects/llc_ddpg_ep_reward.pkl'
TITLE_01 = ('Moving average of reward from training'
            + '\nLunar lander continuous with DDPG')
TITLE_02 = ('Total reward of each episode from training'
            + '\nLunar lander continous with DDPG')
PATH_SAVEFIG_01 = '../images/llc_ddpg_moving_average_reward.png'
PATH_SAVEFIG_02 = '../images/llc_ddpg_reward_each_episode.png'
EPISODES_EVALUATION = 5
FPS = 30


def main():

    # Visualize reward result from training
    ma_reward = pickle.load(open(PATH_REWARD_01, 'rb'))
    plt.plot(ma_reward)
    plt.title(TITLE_01)
    plt.xlabel('Episode')
    plt.ylabel('MA reward')
    plt.tight_layout()
    plt.savefig(PATH_SAVEFIG_01)
    plt.show()

    ep_reward = pickle.load(open(PATH_REWARD_02, 'rb'))
    plt.plot(ep_reward)
    plt.title(TITLE_02)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(PATH_SAVEFIG_02)
    plt.show()

    # Environment
    env = gym.make(ENV)
    num_states, num_actions, bound = environment_spec(env)

    # Model
    model = actor(num_states=num_states, num_actions=num_actions, bound=bound)

    # Make evaluation video
    with imageio.get_writer(PATH_VIDEO, fps=FPS) as video:

        for i in range(EPISODES_EVALUATION):

            # Initialize
            prev_state = env.reset()
            done = False
            screen = env.render(mode='rgb_array')
            video.append_data(screen)

            # Start episode
            while not done:

                # Get state
                prev_state = tf.convert_to_tensor(prev_state)
                prev_state = tf.expand_dims(prev_state, 0)

                # Get action
                action = model(prev_state)
                action = np.clip(action, -bound, bound)
                action = np.squeeze(action)

                # Get next state
                state, _, done, _ = env.step(action)
                screen = env.render(mode='rgb_array')
                video.append_data(screen)
                prev_state = state

            print(f'Evaluation episode {i+1} finished')

    env.close()


if __name__ == "__main__":
    main()
