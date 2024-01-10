# Finite Markov Decision Processes

MDPs = Markov Decision Processes.

Involve
- Delayed reward
- Need to trade off immediate and delayed reward.

Estimate
- Value `q*(s, a)` of each action a in each state s
- Value `v*(s)` of each state given optimal action selections

Returns

Value functions

Bellman equations

**Agent** means the learner and decision maker.

**Environment** means everything outside the agent.

**Trajectory** means the sequence of S0, A0, R1, S1, A1, R2, S2, A2, R3,...

In a finite MDP, the sets of states, actions, and rewards all have a finite number of elements.

**Markov property**, the state must include information about all aspects of the past agent-environment interaction that 
make a difference for the future.

Read from P.50

