# Introduction

Reinforcement learning is learning how to map situations to actions to maximize a numerical reward signal.
- Trial-and-error search
- Delayed reward (Actions may affect not only the immediate reward but also the next situation)

Reinforcement learning is `y = f(x)` and `z`

Trade-off between **exploration** and **exploitation**. The agent has to exploit what it has already experienced in 
order to obtain reward, but it also has to explore in order to make better action selections in the future. Exploration-
exploitation dilemma.

Elements of reinforcement learning
- Policy
- Reward signal
- Value function
- Model of environment (optional)

**Policy** is a mapping from perceived states to actions.

**Reward** is a number that the environment sends to the agent whose goal is to maximize the total reward over the long run.

**Value function** specifies what's good in the long run. The value of a state is the total amount of rewards an agent 
can expect to accumulate over the future. Reward is immediate, while value is long-term. Rewards are given by the 
environment, while values must be estimated. Action choices are made on value judgement. It's the most important to find 
a method to efficiently estimate values.

**Model**, give a state and action, predicts the resultant next state and next reward. **Model-based methods** are RL 
methods using models and planning. **Model-free methods** are trial-and-error learners without planning.

The concepts of **value** and **value function** are key to most of the reinforcement learning methods. Value functions 
are important for efficient search in the space of policies.

## Agent

The system  responsible for interacting with the world and making decisions.

## State

The state of the environment is a description of everything that might change from moment to moment.

## Environment

Anything external to the agent.

Transition, the environment changes from state to state, can be said **stochastic** if it's not necessary that the same 
transition occur every time the agent takes a particular action in a particular state.

But the **probability** of the stochastic transition must remain constant over time. If it's not constant, it's called 
**non-stationary** environments.

**Synchronous** environment is where one state transition occurs for each action the agent takes.

**Asynchronous** environment is where the environment doesn't wait for the agent to take action but instead changes continually.

## Reward

## Policy

People often use the word policy as an abbreviation for stationary policy.

Policies can be stochastic like flipping a weighted coin to decide action.

Read from P.38 1.3.2 planning algorithms, Littman PDF