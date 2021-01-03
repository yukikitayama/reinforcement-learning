import torch
import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pickle
# User defined
from algorithm.PrioritizedReplay import DuelingQNetwork, PERAgent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'device: {device}')

# Parameter
SCORE = '../object/duel_ddqn_per_space_invaders_score.pkl'
MODEL = '../model/duel_ddqn_per_space_invaders_target_model.pth'
VIDEO = '../video/duel_ddqn_per_space_invaders.gif'
SAVEFIG_01 = '../image/duel_ddqn_per_space_invaders_result.png'
ENV = 'SpaceInvaders-v0'
FPS = 30


class GreedyAlgorithm:
    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            action = np.argmax(q_values)
        return action


def visualize_training(score, ma=100, savefig=None):
    ma_score = pd.Series(score).rolling(ma).mean()
    plt.plot(ma_score)
    plt.savefig(savefig)
    plt.show()


def evaluation(agent, episode=3):

    # Gif
    with imageio.get_writer(VIDEO, fps=FPS) as video:

        for i in range(episode):

            state1 = agent.env.reset()
            state2, _, _, _ = agent.env.step(0)
            processed_state1 = agent.process_state(state1)
            processed_state2 = agent.process_state(state2)
            stack_state = agent.stack_state(state=processed_state1,
                                            next_state=processed_state2)

            while True:

                screen = agent.env.render(mode='rgb_array')
                video.append_data(screen)

                action = agent.action_algorithm.select_action(
                    model=agent.target_model,
                    state=stack_state
                )

                # print(f'action: {action}')

                state, _, done, _ = agent.env.step(action)

                if done:
                    break

                processed_state1 = processed_state2
                processed_state2 = agent.process_state(state)
                stack_state = agent.stack_state(state=processed_state1,
                                                next_state=processed_state2)

        agent.env.close()


def main():

    # Environment
    env = gym.make(ENV)

    # Score
    score = pickle.load(open(SCORE, 'rb'))

    # Visualize training
    # visualize_training(score=score)

    # Load trained model
    model = DuelingQNetwork(output_dim=env.action_space.n).to(device)
    model.load_state_dict(torch.load(MODEL))

    # Action algorithm
    action_algorithm = GreedyAlgorithm()

    # Agent
    agent = PERAgent(env=env, device=device, online_model=model, target_model=model,
                     action_algorithm=action_algorithm, tau=0.01, optimizer=None,
                     gamma=0.99, memory=None)

    # Make evaluation video
    evaluation(agent=agent)


if __name__ == '__main__':
    main()
