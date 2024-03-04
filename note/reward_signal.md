# Reward signal

Designing a **reward signal** is a critical part of any application of reinforcement learning. Some problems involve 
goals that are difficult to translate into reward signals.

**Sparse reward**, the agent may wander aimlessly for long periods of time.

A better way to provide guidance to the agent suffering from the sparse reward problem is to leave the reward signal 
alone and instead augment the value-function approximation with an initial guess of what it should be.

**Shaping** technique is an effective approach to the sparse reward problem.

You can modify rewards to shape agent experience without changing the optimal policy
- Scale (positively)
- Shift (add a constant)
- Potential functions



