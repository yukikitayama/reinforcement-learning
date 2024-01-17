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

REad from P.100


