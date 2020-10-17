import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber, Reduction
import numpy as np
from typing import Any, List, Sequence, Tuple
# progress bar
import tqdm
from PIL import Image
print(tf.__version__)

# parameter
ENV = 'CartPole-v0'
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
# small epsilon value for stabilizing division operations?
EPS = np.finfo(np.float32).eps.item()
print(EPS)
NUM_HIDDEN_UNITS = 128
LR = 0.01

# environment
env = gym.make(ENV)
env.seed(SEED)
num_actions = env.action_space.n
state = env.reset()

print('***** Environment info *****')
print('state', env.observation_space)
# [cart position, cart velocity, pole angle, pole velocity at tip]
# https://github.com/openai/gym/wiki/CartPole-v0
print('state example', state)
# Action means, 0: push to the left, 1: push to the right
print('action', env.action_space)
print('reward', env.reward_range)
print('****************************')


# model
class ActorCritic(tf.keras.Model):
    """
    One combined neural network which outputs both actor action probability and
    critic value. Make the network by subclassing tf.keras.Model.
    """

    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()
        self.common = Dense(num_hidden_units, activation='relu')
        self.actor = Dense(num_actions)
        self.critic = Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


model = ActorCritic(num_actions, NUM_HIDDEN_UNITS)


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns state, reward and done flag given an action.

    :param action:
    :return: (state, reward, boolean done)
    """
    state, reward, done, _ = env.step(action)
    return state.astype(np.float32), \
           np.array(reward, np.int32), \
           np.array(done, np.int32)


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    """
    This is a wrapper of OpenAI Gym env.step call as an operation in a Tensor-
    Flow function. This allows it to be included in a callable TensorFlow graph.

    :param action:
    :return:
    """
    return tf.numpy_function(env_step,
                             [action],
                             [tf.float32, tf.int32, tf.int32])


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int) -> List[tf.Tensor]:
    """

    :param initial_state:
    :param model:
    :param max_steps:
    :return: 3 stacked tensors of action probabilities, critic values, and
    rewards after each episode.
    """
    # initialize. dynamic_size enables the size to grow
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    initial_state_shape = initial_state.shape
    state = initial_state
    # start steps in one episode
    for t in tf.range(max_steps):
        # convert state in a tensor from (4,) to (1, 4)
        state = tf.expand_dims(state, 0)
        # get action probability and critic value from model
        action_logits_t, value = model(state)
        # sample next action from the action probability distribution
        # [0, 0] because shape is tf.Tensor[[X]]
        action = tf.random.categorical(logits=action_logits_t, num_samples=1)[0, 0]
        action_probs_t = tf.nn.softmax(logits=action_logits_t)
        # store critic value. squeeze reduces dimention
        values = values.write(index=t, value=tf.squeeze(value))
        # Store log probability of the action
        action_probs = action_probs.write(index=t, value=action_probs_t[0, action])
        # Get next state and reward
        state, reward, done = tf_env_step(action)
        # why below?
        state.set_shape(initial_state_shape)
        # Store reward
        rewards = rewards.write(index=t, value=reward)
        # Check if the episode is over
        if tf.cast(done, tf.bool):
            break

    # stack method returns values in TensorArray to stacked tensor
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True) -> tf.Tensor:
    """
    This converts array of rewards r into array of expected returns G, where
    G is sum of gamma * r.

    :param rewards:
    :param gamma:
    :param standardize:
    :return:
    """
    # n is length of reward array
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    # Start from end of rewards and accumulate reward sum into returns
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    # initialize
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(index=i, value=discounted_sum)
    # After reverse, reverse back again
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + EPS))

    return returns


# Reduction.SUM is scalar sum of weighted losses
huber_loss = Huber(reduction=Reduction.SUM)


def compute_loss(
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor) -> tf.Tensor:
    """
    loss function is actor loss + critic loss. Actor loss is - sum of log policy
    * advantage. Critic loss is huber loss of expected return and value from
    value function.

    :param action_probs:
    :param values:
    :param returns:
    :return:
    """
    # actor
    advantage = returns - values
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    # critic
    critic_loss = huber_loss(values, returns)
    return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=LR)


@tf.function
def train_step(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int) -> tf.Tensor:
    """
    tf.function applies the context to this function so this can be compiled
    into a callable TensorFlow graph, which will be fast. tf.GradientTape does
    automatic differentiation to loss function.

    :return:
    """
    with tf.GradientTape() as tape:
        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(
            initial_state, model, max_steps_per_episode
        )
        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)
        # Convert training data shape to appropriate tensor shape
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]
        ]
        # Calculate loss
        loss = compute_loss(action_probs, values, returns)

    # Compute gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)
    # Apply gradients to model parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


def render_episode(
        env: gym.Env,
        model: tf.keras.Model,
        max_steps: int):
    """
    Makes GIF file from evaluation. This should be run after training.
    :return:
    """
    # initialize
    screen = env.render(mode='rgb_array')
    im = Image.fromarray(screen)
    images = [im]
    state = tf.constant(env.reset(), dtype=tf.float32)
    # start evaluation
    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        # evaluation does not need critic value because it does not update model
        action_probs, _ = model(state)
        action = np.argmax(np.squeeze(action_probs))
        # evaluation does not need reward becase it just outputs how it behave
        # during evaluation
        state, _, done, _ = env.step(action)
        state = tf.constant(state, dtype=tf.float32)
        # Render screen every X steps
        if i % 10 == 0:
            screen = env.render(mode='rgb_array')
            images.append(Image.fromarray(screen))

        if done:
            break

    return images


# training
max_episodes = 10000
max_steps_per_episode = 1000
reward_threshold = 195
running_reward = 0
gamma = 0.99
# tqdm produces progress bar
with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = int(train_step(
            initial_state, model, optimizer, gamma, max_steps_per_episode
        ))
        # What is this?
        running_reward = episode_reward * 0.01 + running_reward * 0.99
        # Set description of the progress bar
        t.set_description(f'Episode {i}')
        # Set postfix (additional stats) with automatic formatting
        t.set_postfix(
            episode_reward=episode_reward,
            running_reward=running_reward
        )
        # Show average episode reward every 10 episodes
        if i % 1 == 0:
            pass # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold:
            break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}')


# evaluation
images = render_episode(env, model, max_steps_per_episode)
image_file = 'cartpole-v0.gif'
images[0].save(
    image_file,
    save_all=True,
    append_images=images[1:],
    loop=0,
    duration=1
)