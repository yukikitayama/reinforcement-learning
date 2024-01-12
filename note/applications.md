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

Read from PDF P.448

