# Coursera Machine Learning Specialization

The **return** depends on the actions you take.

A policy is a function `pi(s) = a`, mapping from states to actions, that tells you what action `a` to take in a given state `s`.

The goal of RL is to find a policy `pi` that tells you what action (`a = pi(s)`) to take in every state (`s`) so as to maximize the return.

Basic components of RL
- States
- Actions
- Rewards
- Discount factor `gamma`
- Return
- Policy `pi`

**Markov decision process (MDP)** says the future state depends on the current state, not depending on how you got to the current state. 

The best possible return from state `s` is `max_a Q(s, a)`

The best possible action in state `s` is the action `a` that gives `max_a Q(s, a)`.
