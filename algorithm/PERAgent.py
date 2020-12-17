import torch
import numpy as np
import random
import time
import tempfile  # Make temporary files and directories
import gc  # Garbage collector interface
from itertools import count
import os


class PERAgent:
    def __init__(self, replay_buffer_fn, value_model_fn, value_optimizer_fn,
                 value_optimizer_lr, max_gradient_norm, training_strategy_fn,
                 evaluation_strategy_fn, n_warmup_batches,
                 update_target_every_steps, tau):
        self.replay_buffer_fn = replay_buffer_fn
        # In practice, value_model_fn = Dueling Deep Q Network
        self.value_model_fn = value_model_fn
        # In practice, value_optimizer_fn is Adam or RMSprop
        self.value_optimizer_fn = value_optimizer_fn
        # Learning rate of optimizer
        self.value_optimizer_lr = value_optimizer_lr
        self.max_gradient_norm = max_gradient_norm
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

    def train(self, make_env_fn, make_env_kargs, seed, gamma, max_minutes,
              max_episodes, gaol_mean_100_reward):
        # Training time
        training_start = time.time()
        last_debug_time = float('-inf')

        # Environment
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        nS = env.observation_space.shape[0]
        nA = env.action_space.n

        # Creates a temporary directory
        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        # This is not a variable, but calling a function of this class
        self.update_network(tau=1.0)
        # Lambda function to set model parameters to optimizer and its learning rate
        self.value_optimizer = self.value_optimizer_fn(self.online_model,
                                                       self.value_optimizer_lr)
        self.replay_buffer = self.replay_buffer_fn()
        # training_strategy is in practice the instance of epsilon greedy algorithm
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        # Seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # What is 5?
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0

        # Training
        for episode in range(1, max_episodes + 1):

            # Initialization for each episode
            episode_start = time.time()
            state = env.reset()
            is_terminal = False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            # Keep counting until break
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
                    self.optimizer_model(experiences)

                # Update target model
                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()

                # Terminate episode
                if is_terminal:
                    # Free memory
                    gc.collect()
                    break

            # Stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            # Evaluate every time one episode is finished
            # Why online model?
            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.evaluation_scores.append(evaluation_score)

            self.save_checkpoint(episode - 1, self.online_model)

            total_step = int(np.sum(self.episode_timestep))

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            # ?
            lst_100_exp_rat = (
                np.array(self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            )
            # ?
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start






    def optimize_model(self, experience):
        idxs, weights, (states, actions, rewards, next_states, is_terminals) = experience

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
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(),
                                       self.max_gradient_norm)
        self.value_optimizer.step()

        # Update priorities in the prioritized experience replay
        priorities = np.abs(td_error.detach().cpu().numpy())
        self.replay_buffer.update(idxs, priorities)

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
