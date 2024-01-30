# Temporal difference

The difference in value estimates as we go from one step to another 

TD falls into value-based, one of the following 3 types

More supervised
- Model-based
- Value-function-based (Model free)
- Policy search
More direct learning

Many researches are focused on value-function-based

`v_t(s) = v_t-1(s) + alpha_t (r_t(s) - v_t-1(s))`

The simplest TD method (**TD(0)**, or **one-step TD**) is `V(S_t) <- V(S_t) + alpha ( R_t+1 + gamma V(S_t+1) - V(S_t) )`

Differences in target of TD(0), MC, DP
- Target of TD(0) `R_t+1 + gamma V(S_t+1)`
- Target of MC `G_t`, but it's estimate, and **sample return** is used instead because expected value of `G_t` is unknown.
- Target of DP `R_t+1 + gamma v_pi(S_t+1)`, but it's estimate because `V(S_t+1)` is used instead because `v_pi(S_t+1)` is unknown. 

and `lim t->inf v_t(s) = v(s)` meaning `v_t(s)` will be actual average of returns when `t` is big enough

Incremental estimate (**outcome based**)

- TD(1) has high variance
- TD(0) has maximum likelihood estimate
- TD(Lambda) interpolates, cleaner estimates, lambda 0.3 to 0.7 empirically good

Combination of Monte Carlo ideas and dynamic programming ideas.
- Learn directly from raw experience without a model of the environment (Like Monte Carlo)
- Update estimate based in part on other learned estimate without final outcome (Like DP)

**Prediction** means estimating the value function `v_pi` for a given policy `pi`

**Control** means finding an optimal policy

**Sample update** is what TD and Monte Carlo do. They look ahead to a sample successor state, and use successor value 
and reward to compute value of original state. **Sample update** differs from **expected update of DP**, which requires 
a complete distribution of all possible successors.

**TD error** is the difference between the better estimate `R_t+1 + gamma V(S_t+1)` and the estimated value `V(s_t)`. 
**Monte Carlo error** is the sum of TD errors.

**Bootstrap** means intending learning a guess from a guess.

Advantage of TD
- Doesn't require a model of the environment, unlike DP
- Doesn't require a model of its reward, unlike DP
- Doesn't require next-state probability distribution, unlike DP
- Can be implemented in an online, incremental fashion (Monte Carlo needs to wait until end of episode)

TD(0) has been proven to converge to `v_pi`

**Batch updating**, updates are made only after processing each complete batch of training data.

TD(0) finds the estimates which are correct for the **maximum-likelihood model of the Markov process**. **Maximum-likelihood 
estimate** of a parameter is the parameter value whose probability of generating the data is the greatest. The estimated 
transition probability from i to j is the fraction of observed transitions from i to j. The expected reward is the average 
rewards over those transitions. Then compute the estimate of value function. This is called **certainty-equivalence estimate**, 
and TD(0) converges to this.

## Learning rate

- Learning rate has to be big enough, so that you can move to what the true value is, no matter where you start
- But the learning rate can't be so big that they don't damp out the noise and actually do a proper job of averaging.

Let `alpha` denote a learning rate.

- `sum alpha = inf` sum of learning rates need to be infinite
- `sum alpha^2 < inf` sum of squared learning rates need to be finite

## Maximum likelihood estimate

The estimate that we get if you use the data to build a model then solve it

## TD Lambda

Both TD(0) and TD(1) have updates based on differences between temporally successive predictions. One algorithm covers both.

- TD(1) has `e(s)` eligibility, but TD(0) doesn't
- TD(1) do for all states, but TD(0) only does for `s_t-1` only the previous state.

## Sarsa

This estimates action-value function `q_pi(s, a)` for the current behavior policy `pi` and for all states `s` and actions `a`

It requires the quintuple of event `(S_t, A_t, R_t+1, S_t+1, A_t+1)`

The update is `Q(S_t, A_t) <- Q(S_t, A_t) + alpha [ R_t+1 + gamma Q(S_t+1, A_t+1) - Q(S_t, A_t) ]`

`Q(s, a)` are arbitrarily initialized for example `Q(s, a) = 0` for all `s`, `a`.

Sarsa can learn during the episode, unlike Monte Carlo

Sara is **on-policy**.

**Expected Sarsa** uses the expected value of `Q(S_t+1, A_t+1)` over all `a`, taking into account how likely each action 
is under the current policy.

`Q(S_t, A_t) <- Q(S_t, A_t) + alpha [ R_t+1 + gamma sum_a pi(a | S_t+1) Q(S_t+1, a) - Q(S_t, A_t) ]`

Given the next state `S_t+1`, this algorithm moves **deterministically** in the same direction as Sarsa moves **in expectation**.

## Maximization bias

When true value `q(s, a)` is 0, but estimated value `Q(s, a)` are uncertain and distributed randomly positive and negative values. 
The maximum of the true value is 0, but the maximum of the estimated values is positive. This is **maximization bias**.

Q-learning could suffer from this maximization bias initially in episodes.

**Double learning** is a way to avoid maximization bias. Have 2 estimates `Q_1(a)` and `Q_2(a)`. `Q_2(A*) = Q_2(argmax_a Q_1(a))` 
This estimate is **unbiased** because `E[Q_2(A*)] = q(A*)`

We have 2 estimates, but only 1 estimate is updated on each play.

## n-step TD method

The method between **Monte Carlo (MC)** method and **one-step temporal difference (TD)** method.

n-step methods enable **bootstrapping** to occur over multiple steps.

n-step methods is intro to **eligibility trace**.

**n-step TD method** is the temporal difference extends over n steps.

**One-step return** means the first reward plus the discounted estimated value of the next state

`G_t:t+1 = R_t+1 + gamma V_t(S_t+1)`

**Two-step retrun** is `G_t:t+2 = R_t+1 + gamma R_t+2 + gamma^2 V_t+1(S_t+2)`

**n-step TD** is the state-value learning algorithm by `V_t+n(S_t) = V_t+n-1(S_t) + alpha [ G_t:t+n - V_t+n-1(S_t) ]`

**Error reduction property** supports that n-step TD method converges to the correct predictions


Read from P.151 6.4 sarsa