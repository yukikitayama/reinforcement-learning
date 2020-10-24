from environment.ForexEnv import ForexEnv
import numpy as np
import matplotlib.pyplot as plt


# Parameter
LENGTH = 30
SEED = 0
PATH_SAVEFIG_01 = '../image/forex_env_test.png'


def print_step(next_state, reward, done):
    return print(f'Next state=[{next_state[0]:,.2f}, {next_state[1]}], '
                 f'Reward={reward:,.2f}, Done={done}')


def main():
    # Test constructor
    env = ForexEnv(episode_length=LENGTH)
    print('*** Test constructor ***')
    print(env)
    print('Size', env.size)
    print('Episode length', env.episode_length)
    print('*** Test constructor ***\n')

    # Test class variables
    print('*** Test class variables ***')
    print('Num states', env.num_states)
    print('Num actions', env.num_actions)
    print('Size', env.size)
    print()

    # Test set_seed method
    env.set_seed(seed_int=SEED)

    # Test reset method
    env.reset()
    print('*** Test reset method ***')
    print(env)
    print('*** Test reset method ***\n')

    # Test step method
    # From no position, make long position
    action = 1
    state, reward, done = env.step(action)
    print('*** Test step method by making long position ***')
    print(env)
    print_step(state, reward, done)
    print()

    # From long position, exit
    action = 0
    state, reward, done = env.step(action)
    print('*** Test step method by closing long position ***')
    print(env)
    print_step(state, reward, done)
    print()

    # From no position to short position
    action = 2
    state, reward, done = env.step(action)
    print('*** Test step method by making short position ***')
    print(env)
    print_step(state, reward, done)
    print()

    # From short to no position
    action = 0
    state, reward, done = env.step(action)
    print('*** Test step method by closing short position ***')
    print(env)
    print_step(state, reward, done)
    print()

    # Test iteration
    print('*** Test iteration and done ***')
    state = env.reset()
    states = []
    done = False
    dones = [done]
    rewards = []
    actions = []
    total_rewards = []
    total_reward = 0
    print('Initial environment', env)
    print('State', state)
    print()
    while not done:

        # Get action
        action = np.random.choice(env.num_actions)

        # Get next state and reward
        next_state, reward, done = env.step(action)
        print(env)
        print_step(next_state, reward, done)
        print()

        # Accumulate reward
        total_reward += reward

        # Store data
        states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        total_rewards.append(total_reward)
        actions.append(action)

    # Visualize
    # print(states)
    rates = []
    positions = []
    for rate, position in states:
        rates.append(rate)
        positions.append(position)

    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(rates, color='#1f77b4')
    plt.title('Rate time series')
    plt.xlabel('Time step')
    plt.ylabel('USD/JPY')
    plt.tight_layout()
    # plt.show()

    plt.subplot(3, 1, 2)
    plt.plot(positions, marker='.', linestyle='--', color='#ff7f0e')
    plt.title('Position time series')
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.yticks(range(0, env.num_actions), ['no position', 'long', 'short'])
    plt.tight_layout()
    # plt.show()

    plt.subplot(3, 1, 3)
    plt.plot(total_rewards, color='#2ca02c')
    plt.title('Total reward time series')
    plt.xlabel('Time step')
    plt.ylabel('Total reward')
    plt.tight_layout()
    plt.savefig(PATH_SAVEFIG_01)
    plt.show()


if __name__ == '__main__':
    main()
