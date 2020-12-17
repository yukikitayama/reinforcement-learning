# Reinforcement Learning

## Overview

This is an repository of reinforcement learning that I'm currently working on.

## Result

* DQN in Lunar Lander discrete action space

![video_06](https://github.com/yukikitayama/reinforcement-learning/blob/master/video/dqn_lunar_lander.gif)

![image_05](https://github.com/yukikitayama/reinforcement-learning/blob/master/image/dqn_lunar_lander_score.png)

* Dueling architecture Double Q Learning with prioritized experience replay after training 4,500 episodes.

![video_01](https://github.com/yukikitayama/reinforcement-learning/blob/master/video/space_invaders_duel_eps4500_short.gif)

* Lunar lander continuous environment by using Deep Deterministic Policy Gradient. (Fail)

![video_05](https://github.com/yukikitayama/reinforcement-learning/blob/master/video/llc_ddpg.gif)

![image_04](https://github.com/yukikitayama/reinforcement-learning/blob/master/image/llc_ddpg_moving_average_reward.png)

* Deep Deterministic Policy Gradient (DDPG) in Pendulum environment and moving average reward

![video_02](https://github.com/yukikitayama/reinforcement-learning/blob/master/video/ddpg_pendulum.gif)

![image_02](https://github.com/yukikitayama/reinforcement-learning/blob/master/image/ddpg_pendulum_v2.png)

* Asynchronous Advantage Actor Critic in CartPole.

![video_03](https://github.com/yukikitayama/reinforcement-learning/blob/master/video/cartpole_a3c.gif)

![image_03](https://github.com/yukikitayama/reinforcement-learning/blob/master/image/reward_a3c_cartpole.png)

* Actor Critic in Mountain Car

![video_04](https://github.com/yukikitayama/reinforcement-learning/blob/master/video/actor_critic_mountaincar.gif)

![image_01](https://github.com/yukikitayama/reinforcement-learning/blob/master/image/reward_duel_space_invaders.png)

## Algorithm

* Deep Q Network (DQN)
* Dueling architecture
* Asynchronous Advantage Actor Critic (A3C)
* Prioritized experience replay (PER)
* Deep Deterministic Policy Gradient (DDPG)

## Environment

* OpenAI Gym
  * Space invaders
  * Lunar lander discrete/continous
  * Cartpole
  * Mountain car

## Tool

* OS: Ubuntu 20.04
* GPU: NVIDIA GeForce RTX 2070
* TensorFlow and PyTorch
* Google Colab
* AWS EC2 Ubuntu 18.04 g4dn.xlarge 1GPU
* AWS Deep Learning AMI 1GPU

## Studying

* Reinforcement Learning An Introduction, Richard S. Sutton and Andrew G. Barto
* Grokking Deep Reinforcement Learning, Miguel Morales
* Coursera Reinforcement Learning Specialization by University of Alberta (https://www.coursera.org/specializations/reinforcement-learning)
* Udacity Deep Reinforcement Learning Nanodegree (https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
* Reading papers from OpenAI Spinning Up key papers (https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
