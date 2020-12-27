import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time

# import pong_utils

# Parameter
ENV = 'PongDeterministic-v4'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
RIGHT = 4
LEFT = 5
NUM_ACTION = 6

# DISCOUNT_RATE = 0.99
DISCOUNT_RATE = 0.995
# EPSILON = 0.1
EPSILON = 0.2  # Epsilon of clipped surrogate function to clip the ratio of current prob divided by old prob
EPSILON_DECAY = 0.999
BETA = 0.01  # c2 coefficient for entropy bonus
BETA_DECAY = 0.995
TMAX = 320
SGD_EPOCH = 4
EPISODE = 10000
EPISODE_MONITOR = 100
LR = 1e-4
SEED = 0


def preprocess_single(image, bkg_color=np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
    return img


def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])


def view_environment(env):
    env.reset()
    _, _, _, _ = env.step(0)
    for _ in range(20):
        frame, _, _, _ = env.step(1)

    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.title('Preprocessed image')
    plt.imshow(preprocess_single(frame), cmap='Greys')
    plt.show()

    print('Shape of preprocessed frame', preprocess_single(frame).shape)


class PolicyTNakae(nn.Module):
    def __init__(self):
        super(PolicyTNakae, self).__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # (80 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1 = 78 / 2 + 1 = 39 + 1 = 40
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, stride=2)
        # (40 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1 = 38 / 2 + 1 = 20
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2)
        # (20 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1 = 18 / 2 + 1 = 10
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
        # (10 + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1 = 8 / 2 + 1 = 5
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        self.size = 32 * 5 * 5
        self.fc1 = nn.Linear(self.size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


class Policy(nn.Module):
    """
    This outputs the probability of moving right, so P(left) = 1 - P(right)
    """
    def __init__(self):
        super(Policy, self).__init__()
        # in_channels=2 is because of 2 stacked frames
        # (80 + 2 * 0 - 1 * (6 - 1) - 1) / 2 + 1 = (80 - 6) / 2 + 1 = 74 / 2 + 1 = 38
        # 80 * 80 * 2 to 38 * 38 * 4
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2, bias=False)
        # (38 + 2 * 0 - 1 * (6 - 1) - 1) / 4 + 1 = (38 - 6) / 4 + 1 = 32 / 4 + 1 = 9
        # 38 * 38 * 4 to 9 * 9 * 16
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))


def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.01):
    """
    Clipped surrogate function.
    """
    # List of discount factors to the power of t
    # First element is gamma^0, followed by gamma^1, gamma^2, and so on
    discount = discount ** np.arange(len(rewards))
    # np.newaxis is used to increase dimension
    # rewards shape becomes (len(discount), len(rewards))
    # First row has rewards each multiplied by discount * gamma^0,
    # second row also has reward each multiplied by discount * gamma^1, and third by discount * gamma^2
    rewards = np.asarray(rewards) * discount[:, np.newaxis]

    # Sum of discounted future rewards for each reward and for each time step
    # rewards_future shape has (len(rewards), len(rewards))
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    # Avoid zero division
    std = np.std(rewards_future, axis=1) + 1.0e-10

    # mean[:, np.newaxis] shape has (len(rewards_future), 1) from mean shape (len(rewards_future), )
    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)

    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)

    # Probability ratio
    ratio = new_probs / old_probs

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, min=1 - epsilon, max=1 + epsilon)

    # Clipped surrogate objective
    # (7) of proximal policy optimization algorithms paper
    clipped_surrogate_objective = torch.min(ratio * rewards, clipped_ratio * rewards)

    # An entropy regularization term, which steers new_policy towards 0.5
    # The form for a binary prediction
    # Adding 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) +
                (1.0 - new_probs) * (torch.log(1.0 - old_probs + 1.e-10)))

    return torch.mean(clipped_surrogate_objective + beta * entropy)


def test_clipped_surrogate():
    rewards = [1, 1, 10]
    discount = 0.995
    discount = discount ** np.arange(len(rewards))
    print('discount\n', discount)
    rewards = np.asarray(rewards) * discount[:, np.newaxis]
    print('rewards\n', rewards, rewards.shape)
    print('rewards[::-1]\n', rewards[::-1])
    print('rewards[::-1].cumsum(axis=0)\n', rewards[::-1].cumsum(axis=0))
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    print('rewards_future\n', rewards_future, rewards_future.shape)

    mean = np.mean(rewards_future, axis=1)
    # Avoid zero division
    std = np.std(rewards_future, axis=1) + 1.0e-10
    print('mean\n', mean, mean.shape)
    print('mean[:, np.newaxis]\n', mean[:, np.newaxis], mean[:, np.newaxis].shape)
    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]
    print('rewards_normalized\n', rewards_normalized, rewards_normalized.shape)


def collect_trajectories(env, policy, tmax, nrand=5):
    n = 1

    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    # Perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = env.step(np.random.choice([RIGHT, LEFT], n))
        fr2, re2, _, _ = env.step(0 * n)

    for t in range(tmax):

        batch_input = preprocess_batch([fr1, fr2])

        probs = policy(batch_input).squeeze().cpu().detach().numpy()

        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)

        probs = np.where(action == RIGHT, probs, 1.0 - probs)

        # Continue game. 0 is do nothing
        fr1, re1, is_done, _ = env.step(action)
        fr2, re2, is_done, _ = env.step(0 * n)

        reward = re1 + re2

        # Store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)

        # Stop if any of the trajectories is done True
        if is_done.any():
            break

    return prob_list, state_list, action_list, reward_list


def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    """
    Returns batch data which has shape (1, 2, 80, 80).
    """
    list_of_images = np.asarray(images)

    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)

    list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color,
                                    axis=-1) / 255.0

    batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
    return torch.from_numpy(batch_input).float().to(device)


def test_preprocess_batch(env):
    env.reset()
    fr1, _, _, _ = env.step(0)
    fr2, _, _, _ = env.step(0)
    batch_input = preprocess_batch([fr1, fr2])
    print(batch_input.size())


def play(env, policy, total_time=2000, preprocess=None, nrand=5):
    env.reset()

    env.render()

    env.step(1)

    # Random steps in the beginning
    for _ in range(nrand):

        env.render()

        fr1, re1, is_done, _ = env.step(np.random.choice([RIGHT, LEFT]))
        fr2, re2, is_done, _ = env.step(0)

    # Store frames in each time step
    anim_frames = []

    for _ in range(total_time):

        env.render()
        time.sleep(0.05)

        frame_input = preprocess_batch([fr1, fr2])
        prob = policy(frame_input)

        # RIGHT = 4, LEFT = 5
        action = RIGHT if random.random() < prob else LEFT
        fr1, _, is_done, _ = env.step(action)
        fr2, _, is_done, _ = env.step(0)

        if preprocess is None:
            anim_frames.append(fr1)
        else:
            anim_frames.append(preprocess(fr1))

        if is_done:
            break

    env.close()


def main():
    env = gym.make(ENV)
    print('Environment', env)
    print('List of available actions:', env.unwrapped.get_action_meanings())

    # Seed
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # View environment
    # view_environment(env)

    # Test clipped_surrogate
    # test_clipped_surrogate()

    # Test preprocess_batch
    # test_preprocess_batch(env)

    # Policy
    policy = Policy().to(device)

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # Random play
    # play(env, policy, total_time=200, preprocess=None, nrand=5)

    # Initialization for training
    epsilon = EPSILON
    beta = BETA
    rewards = []

    # Training
    for episode in range(EPISODE):

        # Initialization for each episode
        state_history = []
        action_history = []
        action_prob_history = []
        reward_history = []
        env.reset()
        total_rewards = 0

        # Random steps in the beginning
        for _ in range(5):
            state1, reward1, is_done, _ = env.step(np.random.choice([RIGHT, LEFT]))
            state2, reward2, is_done, _ = env.step(0)

        for t in range(100000):

            # Preprocess for policy
            batch_input = preprocess_batch([state1, state2])

            # Get action probability and the action
            probs = policy(batch_input).squeeze().cpu().detach().numpy()
            action = np.where(np.random.rand(1) < probs, RIGHT, LEFT)
            probs = np.where(action == RIGHT, probs, 1.0 - probs)

            # Advance the game
            state1, reward1, done, _ = env.step(action)
            state2, reward2, done, _ = env.step(0)  # 0 is no action

            reward = reward1 + reward2

            # Store experience
            state_history.append(batch_input)
            action_history.append(action)
            reward_history.append(reward)
            action_prob_history.append(probs)

            if done:
                total_rewards = sum(reward_history)
                break

        # print(f'Episode: {episode}\tTrajectory ended')

        # Optimize surrogate L with respect to theta with K epochs and minibatch size M <= NT
        for i in range(SGD_EPOCH):

            L = -clipped_surrogate(policy=policy, old_probs=action_prob_history,
                                   states=state_history, actions=action_history,
                                   rewards=reward_history, discount=DISCOUNT_RATE,
                                   epsilon=epsilon, beta=beta)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L
            # print(f'Episode: {episode}\tEpoch: {i+1}')

        # Reduce clipping parameter and entropy regularization coefficient as time goes on
        epsilon *= EPSILON_DECAY
        beta *= BETA_DECAY

        rewards.append(total_rewards)

        if episode % EPISODE_MONITOR == 0:
            print(f'Episode: {episode:d}\t Score: {np.mean(rewards[-100:]):.1f}')


if __name__ == '__main__':
    main()
