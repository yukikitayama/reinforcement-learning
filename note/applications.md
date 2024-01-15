# Applications

Domain knowledge

Representation issue.

Making applications easier and more straightforward is one of the goals of current research in reinforcement learning.

## TD-Gammon

Gerald Tesauro

Uses **TD lambda algorithm** and **nonlinear function approximation** with multilayer artificial neural network, trained 
by backpropagating TD errors.

States and actions of backgammon is huge. It doesn't allow to use the conventional heuristic search methods.

TD-Gammon use standard multilayer ANN to estimate the value function. Input to the network is a backgammon position. 
Output is the estimate of the value of that position. For the input size 198 units, Tesauro tried 
to represent the position in a straightforward way, and keep the number of units relatively small.

TD-Gammon 0.0 was constructed with essentially zero backgammon knowledge, but performed well, so it became testimony to 
the potential of self-play learning methods.

By the following TD-Gammon algorithms, TD-Gammon illustrates the combination of learned value functions and decision-time
search as in heuristic search and MCTS methods.

Jellyfish, Snowie, GNUBackgammon were inspired by TD-Gammon.

## Samuel's checkers player

Samuel's checkers-playing program was widely recognized as a significant achievement in artificial intelligence and 
machine learning.

Used heuristic search methods, a lookahead search from each current position

Samuel conduced his research in th 1950s. He chose checkers instead of chess because of its simplicity and of focusing 
on learning.

**Rote learning**

(temporal-differenc learning)

## Daily-double wagering

Strategy component that IBM WATSON used to play TV quiz show **Jeopardy!**.

## Memory controller

Conducted by Ipek, Mutlu, Martinez, Caruana in the 2000s.

Designed a reinforcement learning memory controller by taking advantage of past scheduling experience and account for 
long-term consequences of scheduling decisions

Scheduler is the reinforcement learning agent. The scheduling agent used **Sarsa** to learn an action-value function.

Improved the speed of program execution

State is the contents of the transaction queue.

Action is the commands to the DRAM system.

Reward is 1 for read or write action, otherwise 0.

It was intended for the learning controller to be implemented on a chip so that learning could occur online while a 
computer is running. But this learning memory controller wasn't committed to physical hardware because of the large cost 
of fabrication.

## Video game play like human

**Deep Q-network (DQN)**, the reinforcement learning agent developed by Mnih et al. It combines Q-learning with a deep 
convolutional ANN.

Atari 2600, arcade video games. Testbeds for developing and evaluation reinforcement learning methods.

Bellemare, Naddaf, Veness, and Bowling developed the publicly available **Arcade Learning Environment (ALE)** to simplify 
Atari 2600 for algorithm learning.

What's impressive is that a good performance was achieved by the same learning system over the widely varying games 
without relying on any game-specific modifications.

Read from P.461.






