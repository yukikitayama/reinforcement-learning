# Dynamic Programming

Algorithms to compute optimal policies given a perfect model of environment as Markov decision processes.

Limited use in reinforcement learning but theoretically important.

It assumes a finite MDP (state, action, reward are finite, and we have probability `p(s', r | s, a)`)

**Policy evaluation (prediction problem)** to compute the state-value function `v_pi` for an arbitrary policy `pi`

**Iterative policy evaluation** the sequence of `{v_k}` can be shown in general converge to `v_pi` as `k -> infinity`

**Expected update** each iteration of iterative policy evaluation updates the value of every state once to produce the 
new approximate value function. It's called **expected** because the updates are based on an expectation over all 
possible next states instead of a sample next state.

DP algorithms have the array of `v_k(s)` and typically iterativelly updated in-place.

## Policy improvement theorem

**Policy improvement** is the process of making a new policy that improves on an original policy, by making it greedy 
with respect to the value function of the original policy.

## Policy iteration

Sequence of policy evaluations and policy iterations to converge to an optimal policy and the optimal value function in 
a finite number of iterations.

## Value iteration

[placeholder]

## Asynchronous DP

A drawback of DP methods is that it involves operations over the entire state set of the MDP, and it's expensive if the 
state set is very large.

The values of some states may be updated several times before the values of others are updated once.




REad from P.107




