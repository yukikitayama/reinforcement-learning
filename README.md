# Reinforcement Learning

## Overview

This is an ongoing self-learning repository of reinforcement learning. My curret goal is to make a result that I can be happy with from OpenAI Gym Atari Space Invaders environment. Usually I'm working on during weekend when I have time.

## Results

* Dueling architecture Double Q Learning with prioritized experience replay after training 4,500 episodes.

![video_01](https://github.com/yukikitayama/reinforcement-learning/blob/master/videos/space_invaders_duel_eps4500_short.gif)

* Deep Deterministic Policy Gradient (DDPG) in Pendulum environment

![video_02](https://github.com/yukikitayama/reinforcement-learning/blob/master/videos/ddpg_pendulum.gif)

* Asynchronous Advantage Actor Critic in CartPole.

![video_03](https://github.com/yukikitayama/reinforcement-learning/blob/master/videos/cartpole_a3c.gif)

* Moving average rewards from the above result (Unsatisfactory...)

![image_01](https://github.com/yukikitayama/reinforcement-learning/blob/master/images/reward_duel_space_invaders.png)

* Moving average reward from DDPG Pendulum

![image_02](https://github.com/yukikitayama/reinforcement-learning/blob/master/images/ddpg_pendulum_v2.png)

* Below is moving average reward from A3C. Looks nice, but it's CartPole.

![image_03](https://github.com/yukikitayama/reinforcement-learning/blob/master/images/reward_a3c_cartpole.png)

## Algorithms

* Deep Q Network (DQN)
* Dueling architecture
* Asynchronous Advantage Actor Critic (A3C)
* Prioritized experience replay (PER)
* Deep Deterministic Policy Gradient (DDPG)

## Tools

* Google Colab
* AWS EC2 Ubuntu 18.04 g4dn.xlarge 1GPU
  * Set up Jupyter Lab and run Jupyter Notebooks
* AWS Deep Learning AMI 1GPU

## Studying approach

* Reinforcement Learning An Introduction, Richard S. Sutton and Andrew G. Barto
* Coursera Reinforcement Learning Specialization by University of Alberta (https://www.coursera.org/specializations/reinforcement-learning)
* Reading papers from OpenAI Spinning Up key papers (https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
* Reading blogs.
