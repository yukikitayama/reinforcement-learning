# Generalized Markov Decision Processes.

All models have an optimal value function and an optimal policy.

Value iteration converges to the optimal value function.

State space can be infinite

Summary operators. They need to be **non-expansion**.
- Summary operator `x` is for max over actions
  - Summarizes the value of a finite set of actions for each state.
- Summary operator `+` is for expectation over next states
  - Summarizes the value of a finite set of next states for each state-action pair

Dynamic-programming operator `H` for a generalized MDP. Dynamic-programming operator `K` acts on Q functions. 
They are **contraction mapping**. `H` brings two value functions closer together. `K` brings two Q functions closer together.

Read Littman 1996 from P.56
