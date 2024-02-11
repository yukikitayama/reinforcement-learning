# n-step Bootstrapping

One-step TD and Monte Carlo are at the extreme end. Methods that involve an intermediate amount of bootstrapping are 
important because they will typically perform better than either extreme.

n-step methods look ahead to the next `n` rewards, states, and actions. All n-step methods involve a delay of `n` time 
steps before updating.

A further drawback is that they involve more computation per time step.

## n-step Sarsa

All start and end with an action rather than a state like n-step TD.

In n-step Expected Sarsa, if `s` is terminal, then its expected approximate value is defined to be 0.

## n-step Off-policy Learning

**Off-policy learning** means to learn the value function for `pi` while following another policy `b`. `pi` is the greedy 
policy for the current action-value function estimate, `b` is exploratory policy perhaps **epsilon-greedy**.

They say that this is simple and conceptually clear, but not the most efficient.






