import gym
import torch
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import pickle
# User defined
from algorithm.dqn_agent import Agent


# Parameter
ENV = 'LunarLander-v2'
MODEL = '../model/checkpoint_dqn_ll.pth'
VIDEO = '../video/dqn_lunar_lander.gif'
SCORE = '../object/dqn_lunar_lander_score.pkl'
SAVEFIG = '../image/dqn_lunar_lander_score.png'
EPISODE = 10
MAX_STEP = 400
FPS = 30
SEED = 0


def visualize_score(score, ma):
    ma_score = pd.Series(score).rolling(ma).mean()
    upper = pd.Series(score).rolling(ma).quantile(0.9, interpolation='linear')
    lower = pd.Series(score).rolling(ma).quantile(0.1, interpolation='linear')
    plt.plot(ma_score)
    plt.fill_between(x=range(len(upper)),
                     y1=upper,
                     y2=lower,
                     alpha=0.3)
    plt.title(f'DQN Lunar Lander Discrete\n'
              f'{ma}-episode moving average of scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig(SAVEFIG)
    plt.show()


def evaluation(env, agent):

    with imageio.get_writer(VIDEO, fps=FPS) as video:

        for i in range(EPISODE):

            # Initialize
            state = env.reset()
            screen = env.render(mode='rgb_array')
            video.append_data(screen)

            # Start episode
            for j in range(MAX_STEP):

                action = agent.act(state)
                state, reward, done, _ = env.step(action)

                screen = env.render(mode='rgb_array')
                video.append_data(screen)

                if done:
                    break

        env.close()

    print('Saved video')


def main():
    # Environment
    env = gym.make(ENV)
    env.seed(SEED)

    # Agent
    agent = Agent(state_size=8, action_size=4, seed=0)

    # Load trained network
    agent.qnetwork_local.load_state_dict(torch.load(MODEL, map_location=torch.device('cuda:0')))

    # Score result
    score = pickle.load(open(SCORE, 'rb'))

    # Visualize score
    visualize_score(score=score, ma=100)

    # Make evaluation video
    evaluation(env=env, agent=agent)


if __name__ == '__main__':
    main()
