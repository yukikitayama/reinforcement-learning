from environment.ForexEnv import ForexEnv
import numpy as np
import matplotlib.pyplot as plt


# Parameter
LENGTH = 30
SEED = 0
PATH_SAVEFIG_01 = '../image/experiment/forex_env_test.png'
POSITION_STATE_DIM = 3


def print_step(action, curr_state, next_state, reward, done):
    return print(
        f'Action={action}, '
        f'Current state=[{curr_state[0]:,.2f}, {curr_state[1]}], '
        f'Next state=[{next_state[0]:,.2f}, {next_state[1]}], '
        f'Reward={reward:,.2f}, Done={done}'
    )


def main():
    # Test constructor
    env = ForexEnv(length=LENGTH)
    print('*** Test constructor ***')
    print(env)
    print('Size', env.size)
    print('Length', env.length)
    print()

    # Test class variables
    print('*** Test class variables ***')
    print('Num states', env.num_states)
    print('Num actions', env.num_actions)
    print('Size', env.size)
    print()

    # Test set_seed method
    env.set_seed(seed_int=SEED)
    print('*** Test set_seed ***')
    print(env)
    print('Seed', SEED)
    print()

    # Test reset method
    reset_state = env.reset()
    print('*** Test reset method ***')
    print(env)
    print('Reset state', reset_state)
    print()

    # Test step method
    curr_state = reset_state
    # Open short position
    action = 1
    next_state, reward, done = env.step(action)
    print('*** Test step method by opening short position ***')
    print(env)
    print_step(action, curr_state, next_state, reward, done)
    print()

    # Close short position
    curr_state = next_state
    action = 3
    next_state, reward, done = env.step(action)
    print('*** Test step method by closing short position ***')
    print(env)
    print_step(action, curr_state, next_state, reward, done)
    print()

    # Open long position
    curr_state = next_state
    action = 2
    next_state, reward, done = env.step(action)
    print('*** Test step method by opening long position ***')
    print(env)
    print_step(action, curr_state, next_state, reward, done)
    print()

    # Close long position
    curr_state = next_state
    action = 3
    next_state, reward, done = env.step(action)
    print('*** Test step method by closing long position ***')
    print(env)
    print_step(action, curr_state, next_state, reward, done)
    print()

    # Test done by iterating
    print('*** Test done by iterating ***')
    curr_state = env.reset()
    states = []
    done = False
    dones = [done]
    rewards = []
    actions = []
    total_rewards = []
    total_reward = 0
    print('Initial environment', env)
    print('Initial state', curr_state)
    print()
    while not done:

        # Get action
        action = np.random.choice(env.num_actions)

        # Get next state and reward
        next_state, reward, done = env.step(action)
        print(env)
        print_step(action, curr_state, next_state, reward, done)
        print()

        # Accumulate reward
        total_reward += reward

        # Store data
        states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        total_rewards.append(total_reward)
        actions.append(action)

        # Iterate state
        curr_state = next_state

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
    plt.yticks(range(0, POSITION_STATE_DIM), ['Short', 'Flat', 'Long'])
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
