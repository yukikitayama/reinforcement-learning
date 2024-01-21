# Monte Carlo

- Method to estimate value functions and find optimal policies
- Not assume complete knowledge of environment
- Require a model to generate sample transition (actual/simulated experience)
- Update value function and optimal policy only on episode completion (episode-by-episode, not step-by-step online)
- Averaging sample returns

The value of a state is the expected return starting from the state (Expected cumulative future discounted reward)

The idea of Monte Carlo method is to take average over the returns observed after visits to a particular state.

- First-visit MC method
- Every-visit MC method

## Backup diagram

DP diagram shows all possible transitions, and only one-step transitions

Monte Carlo diagram shows only those sampled on the one episode, and goes all the way to the end of episode.

## Action values

Without a model, state values alone are not sufficient

We need to estimate the value of each action for the values to be useful in suggesting a policy

**Exploring starts** is to specify that the episodes start in a state-action pairr. 

Read from P.115