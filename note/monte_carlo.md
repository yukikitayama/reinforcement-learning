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

Read from P.115