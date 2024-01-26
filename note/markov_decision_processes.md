# Markov Decision Processes

MDPs = Markov Decision Processes.

MDP is a model of an agent interacting synchronously with its environment.

The assumption is, no uncertainty about the agent's current state, though there may be uncertainty about the effect of 
an agent's actions.

The following things defines the Markov Decision Processes.
- States
- Actions
- Model
- Reward
- Markov property
- Stationary

This Markov decision processes defines a problem, and **policy** is the solution to the problem.

What we want to achieve is to find out, given a problem of Markov Decision Processes, how we can get to the optimal policy.

---

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

## Markov property

the state must include information about all aspects of the past agent-environment interaction that 
make a difference for the future.

**Reward hypothesis** - The agent's goal is to maximize the total amount of reward (scalar signal) it receives, not immediate reward, but cumulative reward in 
the long run. 

## Reward (reward function)

Can be described as `R(s)`, `R(s, a)`, or `R(s, a, s')`, but they are mathematically equivalent.

It's critical the rewards we set up truly indicate what we want to achieve. For example, in chess-playing agent, the 
reward should be given only for actually winning, not for achieving subgoals like taking its opponent's pieces. 

If the model supplies a small negative reward, and the terminal state gives a positive reward, it encourages the agent 
to end the game. But minor changes matter.

If you design MDP, reward is a domain knowledge. You need to think carefully how you set a reward. Reward tells you how 
to get to the end

### Sequence of rewards

It assumes the MDP has the infinite time horizons. If it's a finite time horizon, you could have multiple policies for 
one state. Without infinite horizon assumption, you will lose the stationarity in your policy.

Utility of sequences, if `U(s0, s1, s2, ...) > U(s0, s'1, s'2, ...)`, then `U(s1, s2, ...) > U(s'1, s'2, ...)` **Stationary preferences**

`U(s0, s1, s2, ...) = sum_{inf}_{t = 0} R(s_t) = sum_{inf}_{t = 0} gamma^t R(s_t) 0 <= gamma < 1`

`<= sum_{inf}_{t = 0} gamma^t R_max = R_max / (1 - gamma)` **Geometric series**

Discounted reward give us the geometric series. It allows us to add infinite number of numbers but gives us the finite number.

## Utility

The expected long-term rewards. It's different from reward, which is an immediate outcome.

We seek to maximize the **expected return** `G_t`, function of the reward sequence.

**Episode**, subsequences of the agent-environment interaction. Each episode ends in the **terminal state**, and reset 
to starting state. The next episode begins independently of the previous episode. **Episodic task** means the tasks in 
episodes.

`S` is nonterminal state. `S+` is the terminal state. `T` is the time of termination.

Precisely, `S_{t, i}` the state representation at time `t` of episode `i`. But sometimes `S_{t}` and `S_{t, i}` are used 
interchangeably.

**Continuing task**, some agent-environment interaction does't break and on-going. The terminal time and the return are 
infinite and problematic.

**Discounting**, in this approach, the agent takes actions to maximize the expected **discounted return** with 
**discounting rate** `0 <= gamma <= 1`.
- If `gamma < 1`, the infinite sum has a finite value as long as the reward sequence is bounded
- If `gamma = 0`, the agent only maximize the immediate reward
- As `gamma` approaches 1, the return objective takes future rewards into account more strongly and the agent becomes 
more farsighted.
The return is a sum of an infinite number of terms, but it's still finite if the reward is nonzero and constant and `gamma < 1`.

When we introduce **absorbing state** at the episode termination producing 0 rewards infinitely, episodic task becomes 
continuing task.

**Value functions**, functions of states (or of state-action pairs) that estimates **how good** it is for the agent to 
be in a given state (or how good it is to perform a given action in a given state). How good is defined in the expected 
return.

## Policy

**Policy** is a mapping from states to probabilities of selecting each action `pi(a | s)`

`pi(s) -> a`

`pi*` is the optimal policy. It maximizes the long-term expected reward.

**Stationary policy** `pi: S -> A` specifies, for each state, an action to be taken.

`v_pi(s)` is the value function of a state `s` under a policy `pi`. It's the expected return when starting in `s` and 
following `pi` thereafter. The value of the terminal state is always 0. `v_pi` is **state-value function for policy pi**.

`q_pi(s, a)` is the value of taking action `a` in state `s` under a policy `pi`. It's the expected return starting from s, 
taking the action a, and thereafter following policy pi. `q_pi` is **action-value function for policy pi**.

**Bellman equation for v_pi** is expression of a relationship between the value of a state and the values of its 
successor states.

Solving a reinforcement learning task means finding a policy that achieves a lot of reward over the long run.

**Optimal policy** `pi_*` always exists which is better than or equal to all other policies, and it has the **optimal 
state-value function** `v_*(s)`, and it has the **optimal action-value function** `q_*(s, a)`.

**Bellman optimality equation** expresses the fact that the value of a state under an optimal policy must equal the 
expected return for the best action from that state.

The optimal action-value function allows optimal actions to be selected without having to know anything about possible 
successor states and their values.

In practice, optimal policies can be generated only with extreme computational cost. It's usually not possible to simply 
compute an optimal policy by solving the Bellman optimality equation.

**Tabular case** is the case where the tasks are small, finite state sets, and it's possible to form the approximations
using arrays or tables, and the corresponding methods are called **tabular methods**.

But in many cases, there are far more states than table entries, so we need to use compact parameterized function 
representation and the functions must be approximated.

## Model (Transition model, transition function, state transition function)

Rules of the game we are playing, the physics of the world.

`T(s, a, s') ~ Pr(s' | s, a)` probability that the agent will visit `s'` when the agent are at `s` and take action `a`.

## V and Q

Value function (V) and action-value function (Q) looks similar but the action-value function is more useful in reinforcement learning.

Because we will know Q by taking expectation from the experience, and we don't need to know reward function and transition function in advance.

But value function requires us to know reward function and transition function in advance.

## Value function

For how to compute a value function, the value function for policy `pi` is the unique solution of a set of simultaneous linear equations, one for each state 
`s`. The system of linear equations can be solved by Gaussian elimination or any of a number of other methods.

## Q function

`Q(s, a)`, state-action value function

## Value iteration

**Bellman error magnitude** is the maximum difference between two successive value functions.

## Policy iteration

The initial value function `v_0` is defined to be zero for all states

Policy iteration can converge in fewer iterations than value iteration

But the increased speed of convergence of policy iteration can be more than offset by the increased computation per 
iteration.



