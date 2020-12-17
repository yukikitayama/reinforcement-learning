import torch
import gym
from gym import wrappers
import numpy as np
import os
import tempfile
import time
import random
from itertools import count
import gc


class GreedyStrategy:
    """
    This is used for evaluation
    """
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)


class EGreedyExpStrategy:
    """
    Exponentially decaying epsilon-greedy exploration strategy. This is used for training.
    """
    def __init__(self, init_epsilon, min_epsilon, decay_steps=20000):
        """
        decay_steps is the range when epsilon exponentially decays.
        """
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        # First makes the base values which starts from 1.0 and ends at 0.0 exponentially
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        # Then adjust the start and end to be init_epsilon and min_epsilon respectively without changing exponential shape
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        # Use t to get an epsilon from epsilons list. Also used to fix epsilon at min.
        self.t = 0
        # ?
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state):
        # Set default value
        self.exploratory_action_taken = False

        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))

        # Decay epsilon
        self._epsilon_update()
        # The selected action is different from the action from max q values means exploration
        self.exploratory_action_taken = action != np.argmax(q_values)

        return action


def get_make_env_fn(**kargs):

    def make_env_fn(env_name, seed=None, render=None, record=False,
                    unwrapped=False, monitor_mode=None,
                    inner_wrappers=None, outer_wrappers=None):
        mdir = tempfile.mkdtemp()
        env = None
        if render:
            try:
                env = gym.make(env_name, render=render)
            except:
                pass
        if env is None:
            env = gym.make(env_name)
        if seed is not None: env.seed(seed)
        env = env.unwrapped if unwrapped else env
        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)
        env = wrappers.Monitor(
            env, mdir, force=True,
            mode=monitor_mode,
            video_callable=lambda e_idx: record) if monitor_mode else env
        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)
        return env

    return make_env_fn, kargs


class PERAgent:
    def __init__(self, online_model, target_model, gamma, value_optimizer,
                 replay_buffer, tau, training_strategy, evaluation_strategy,
                 n_warmup_batches, update_target_every_steps):
        self.online_model = online_model
        self.target_model = target_model
        self.gamma = gamma
        self.value_optimizer = value_optimizer
        self.replay_buffer = replay_buffer
        self.tau = tau
        self.training_strategy = training_strategy
        self.evaluation_strategy = evaluation_strategy
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps

        self.episode_reward = []
        self.episode_timestep = []
        self.episode_exploration = []
        self.checkpoint_dir = tempfile.mkdtemp()
        self.evaluation_scores = []

    def optimize_model(self, experiences):
        idxs, weights, (states, actions, rewards, next_states, is_terminals) = experiences

        # Set to device
        weights = self.online_model.numpy_float_to_device(weights)

        batch_size = len(is_terminals)

        # Target
        # Double learning
        # online model get the action index
        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        # target model calculates q values
        q_sp = self.target_model(next_states).detach()
        # Select a q value by the action index
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))

        # Estimate
        q_sa = self.online_model(states).gather(1, actions)

        # Loss
        td_error = q_sa - target_q_sa
        # Weighted importance-sampling
        value_loss = (weights * td_error).pow(2).mul(0.5).mean()

        # Gradient decsent
        self.value_optimizer.zero_grad()
        value_loss.backward()

        # Gradient clip
        # torch.nn.utils.clip_grad_norm_(self.online_model.parameters(),
        #                                self.max_gradient_norm)

        self.value_optimizer.step()

        # Update priorities in the prioritized experience replay
        priorities = np.abs(td_error.detach().cpu().numpy())
        self.replay_buffer.update(idxs, priorities)

    def update_network(self, tau=None):
        """
        Polyak averaging to update target model weights.
        """
        tau = self.tau if tau is None else tau

        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            # Use torch inplace function to update target model weights
            target.data.copy_(mixed_weights)

    def interaction_step(self, state, env):
        """
        This is used during training to get the next state and terminal flag from environment. This also collects
        reward, increments time step, and collects boolean of whether action is exploitation or exploration.
        """
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, info = env.step(action)
        # ?
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        # ?
        is_failure = is_terminal and not is_truncated

        experience = (state, action, reward, new_state, float(is_failure))

        # Store data
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)

        return new_state, is_terminal

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):

            s  = eval_env.reset()
            d = False
            rs.append(0)

            for _ in count():

                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)

                rs[-1] += r

                if d:
                    break

        return np.mean(rs), np.std(rs)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))

    def train(self, make_env_fn, make_env_kargs, seed, max_episodes,
              goal_mean_100_reward):
        training_start = time.time()

        env = make_env_fn(**make_env_kargs, seed=seed)
        nS = env.observation_space.shape[0]
        nA = env.action_space.n

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        for episode in range(1, max_episodes + 1):

            state = env.reset()
            is_terminal = False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():

                state, is_terminal = self.interaction_step(state, env)

                # ?
                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches

                # Update online model
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    idxs, weights, samples = experiences
                    # load method makes sure input is torch tensor and set to device
                    experiences = self.online_model.load(samples)
                    experiences = (idxs, weights) + (experiences,)
                    # Update online model weights
                    self.optimize_model(experiences)

                # Update target model
                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()

                # Terminate episode
                if is_terminal:
                    # Free memory
                    gc.collect()
                    break

            # Online model?
            self.save_checkpoint(episode - 1, self.online_model)

            # Online model?
            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.evaluation_scores.append(evaluation_score)

            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])

            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward

            training_is_over = reached_goal_mean_reward

            if training_is_over:
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break

        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        print('Training complete.')

        env.close()
        del env

        return final_eval_score
