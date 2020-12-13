import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import pickle
# User defined
from dqn_agent import Agent
plt.style.use('default')


# Parameter
ENV = 'LunarLander-v2'
STATE_SIZE = 8
ACTION_SIZE = 4
SEED = 0
N_EPISODES = 2000
MAX_T = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
MAXLEN = 100
THRESHOLD = 200.0
MODEL = '../model/checkpoint_dqn_ll.pth'
PATH_SCORE = '../object/dqn_lunar_lander_score.pkl'


def dqn(n_episodes, max_t, eps_start, eps_end, eps_decay, env, agent):
    # Initialization for training
    scores = []
    scores_window = deque(maxlen=MAXLEN)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):

        # Initialization for each episode
        state = env.reset()
        score = 0

        for t in range(max_t):

            action = agent.act(state, eps)

            next_state, reward, done, _ = env.step(action)

            # Store experience and update network
            agent.step(state, action, reward, next_state, done)

            state = next_state

            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)

        # Update epsilon
        eps = max(eps_end, eps * eps_decay)

        # Monitor
        print(f'\rEpisode {i_episode}\t'
              f'Average Score: {np.mean(scores_window):.2f}\t'
              f'Epsilon: {eps:.3f}',
              end='')

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\t'
                  f'Average Score: {np.mean(scores_window):.2f}\t'
                  f'Epsilon: {eps:.3f}')
        if np.mean(scores_window) >= THRESHOLD:
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\t'
                  f'Average Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), MODEL)
            print('Saved model')
            break

    return scores


def plot_score(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def main():
    # GPU
    print('Is GPU available?', torch.cuda.is_available())
    print()

    # Environment
    env = gym.make(ENV)
    env.seed(0)
    print('Environment:', env)
    print('State shape:', env.observation_space.shape)
    print('Number of actions:', env.action_space.n)
    print()

    # DQN agent
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=SEED)

    # Training
    scores = dqn(n_episodes=N_EPISODES, max_t=MAX_T,
                 eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
                 env=env, agent=agent)

    # Save score
    print(scores[-5:])
    pickle.dump(scores, open(PATH_SCORE, 'wb'))
    print('Saved score')

    # Visualize training result
    plot_score(scores)


if __name__ == '__main__':
    main()
