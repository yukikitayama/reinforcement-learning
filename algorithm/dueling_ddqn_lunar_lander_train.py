import gym
import torch
import torch.optim as optim
# User defined
from algorithm.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from algorithm.per_agent import GreedyStrategy, EGreedyExpStrategy, PERAgent
from algorithm.DuelingQNetwork import DuelingQNetwork, DuelingQ


# Parameter
ENV = 'LunarLander-v2'
GAMMA = 0.99
LR = 5e-4
# print('Learning rate for optimizer', LR)
SEED = 0
STATE_SIZE = 8
ACTION_SIZE = 4
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 100000
MAX_SAMPLES = 20000
BATCH_SIZE = 64
RANK_BASED = False
ALPHA = 0.6
BETA0 = 0.1
BETA_RATE = 0.99995
N_WARMUP_BATCHES = 5
UPDATE_TARGET_EVERY_STEPS = 4
TAU = 0.1
THRESHOLD = 200.0
CHECKPOINT_DIR = '/home/yuki/PycharmProjects/reinforcement-learning/model'
# MAX_EPISODES = 20000
MAX_EPISODES = 2000
# MAX_EPISODES = 1000


def main():
    # Environment
    env = gym.make(ENV)
    print('Environment\n', env)
    print()

    # Network
    online_model = DuelingQ(state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=SEED)
    target_model = DuelingQ(state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=SEED)
    print('Online model\n', online_model)
    print()

    # Optimizer
    optimizer = optim.Adam(online_model.parameters(), lr=LR)
    print('Optimizer\n', optimizer)
    print()

    # Epsilon greedy algorithm
    training_strategy = EGreedyExpStrategy(init_epsilon=EPS_START,
                                           min_epsilon=EPS_END,
                                           decay_steps=EPS_DECAY)
    print('Training strategy\n', training_strategy)
    print()

    # Greedy algorithm
    evaluation_strategy = GreedyStrategy()
    print('Evaluation strategy\n', evaluation_strategy)
    print()

    # Prioritized experience replay buffer
    buffer = PrioritizedReplayBuffer(
        max_samples=MAX_SAMPLES, batch_size=BATCH_SIZE, rank_based=RANK_BASED,
        alpha=ALPHA, beta0=BETA0, beta_rate=BETA_RATE
    )
    print('Replay memory\n', buffer)
    print()

    # Agent for Dueling DDQN with Prioritized Experience Replay
    agent = PERAgent(
        online_model=online_model,
        target_model=target_model,
        gamma=GAMMA,
        value_optimizer=optimizer,
        replay_buffer=buffer,
        tau=TAU,
        training_strategy=training_strategy,
        evaluation_strategy=evaluation_strategy,
        n_warmup_batches=N_WARMUP_BATCHES,
        update_target_every_steps=UPDATE_TARGET_EVERY_STEPS,
        checkpoint_dir=CHECKPOINT_DIR
    )
    print('Agent\n', agent)
    print()

    # Training
    score = agent.train(
        env=env,
        seed=SEED,
        max_episodes=MAX_EPISODES,
        goal_mean_100_reward=THRESHOLD
    )


if __name__ == '__main__':
    main()
