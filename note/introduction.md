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

Read from p.7 1.4 limitations and scope
 