import gym
import torch
from torch.distributions import Categorical
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import pickle
# User defined
from algorithm.ppo_pong import Policy, PPOAgent

# Parameter
ENV = 'PongDeterministic-v4'
MODEL = '../model/ppo_pong.pth'
# VIDEO = '../video/ppo_pong.gif'
VIDEO = '../video/ppo_pong.mp4'
SCORE = '../object/ppo_pong_score.pkl'
SAVEFIG_01 = '../image/ppo_pong_score.png'
SAVEFIG_02 = '../image/ppo_pong_time.png'
SAVEFIG_03 = '../image/ppo_pong_loss.png'
SAVEFIG_04 = '../image/ppo_pong_all.png'
TIME = '../object/ppo_pong_time.pkl'
LOSS = '../object/ppo_pong_loss.pkl'
EPISODE = 3
MAX_STEP = 10000
FPS = 30
SEED = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device', device)
RIGHT = 4
LEFT = 5


def visualize_result(data, ma, ylabel, savefig):
    ma_data = pd.Series(data).rolling(ma).mean()
    upper = pd.Series(data).rolling(ma).quantile(0.9, interpolation='linear')
    lower = pd.Series(data).rolling(ma).quantile(0.1, interpolation='linear')
    plt.plot(ma_data)
    plt.fill_between(x=range(len(upper)),
                     y1=upper,
                     y2=lower,
                     alpha=0.3)
    plt.title(f'Proximal Policy Optimization Pong\n'
              f'{ma}-episode moving average of {ylabel.lower()}')
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()


def visualize_all(data_list, name_list, color_list, ma, savefig):
    plt.suptitle('Proximal Policy Optimization Pong\n100-episode moving averages')
    for i, data in enumerate(data_list):

        ma_data = pd.Series(data).rolling(ma).mean()
        upper = pd.Series(data).rolling(ma).quantile(0.9, interpolation='linear')
        lower = pd.Series(data).rolling(ma).quantile(0.1, interpolation='linear')

        plt.subplot(len(data_list), 1, i+1)
        plt.plot(ma_data, color=color_list[i])
        plt.fill_between(x=range(len(upper)), y1=upper, y2=lower, alpha=0.3, color=color_list[i])
        plt.ylabel(name_list[i])
        plt.grid()

        if i == 2:
            plt.xlabel('Episode')

    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()


def evaluation(agent, episode, max_step):

    # gif
    # with imageio.get_writer(VIDEO, fps=FPS) as video:

    # mp4
    with imageio.get_writer(VIDEO, format='FFMPEG', fps=FPS, macro_block_size=None) as video:

        for i in range(episode):

            state1, state2 = agent.random_beginning()

            # Start episode
            for j in range(max_step):

                batch_input = agent.pre_process(state1, state2)

                with torch.no_grad():
                    logits = agent.policy(batch_input)

                m = Categorical(logits=logits)
                action = int(m.sample().cpu().numpy()[0])
                action = agent.convert_action_for_env(action)

                state1 = state2
                state2, reward, done, _ = agent.env.step(action)

                screen = agent.env.render(mode='rgb_array')
                video.append_data(screen)

                if done:
                    break

        agent.env.close()

    print('Saved video')


def main():
    # Environment
    env = gym.make(ENV)
    env.seed(SEED)

    # Visualize score
    score = pickle.load(open(SCORE, 'rb'))
    # visualize_result(data=score, ma=100, ylabel='Score', savefig=SAVEFIG_01)

    # Visualize time step
    timesteps = pickle.load(open(TIME, 'rb'))
    # visualize_result(data=timesteps, ma=100, ylabel='Time steps', savefig=SAVEFIG_02)

    # Visualize loss
    losses = pickle.load(open(LOSS, 'rb'))
    # visualize_result(data=losses, ma=100, ylabel='Loss', savefig=SAVEFIG_03)

    # Visualize all
    data_list = [score, timesteps, losses]
    name_list = ['Score', 'Time step', 'Loss']
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    # visualize_all(data_list=data_list, name_list=name_list, color_list=color_list, ma=100, savefig=SAVEFIG_04)

    # Load trained network
    policy = Policy().to(device)
    policy.load_state_dict(torch.load(MODEL, map_location=device))

    # Agent
    agent = PPOAgent(env=env, policy=policy, gamma=0.99, epsilon=0.1, device=device, nrand=5)

    # Make evaluation video
    evaluation(agent=agent, episode=5, max_step=300)


if __name__ == '__main__':
    main()
