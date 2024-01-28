# Learning to predict by the methods of temporal differences

The paper is meant to prove the convergence and optimality of TD methods for important special cases.

Learning to predict == prediction learning,

Temporal-difference (TD) methods, incremental learning procedures for prediction problems.

TD methods are by the error (differences) between temporally successive predictions

TD methods are 
1. more incremental and easier to compute.
2. more efficient use of their experience.

As methods for efficiently learning to predict **arbitrary events**, not just goal-related ones.

The paper is concerned with multi-step prediction problems

Notations
- `x_t` state
- `z` reward
- `w` modifiable parameters or weights for prediction
- `P_t` estimate of `z`, can be `P(x_t, w)`

Read from 2.1 single-step...