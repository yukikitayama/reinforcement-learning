# Eligibility trace

**Eligibility trace** updates the various estimates up to the beginning of the episode, to different degrees, fading 
with recency. The fading strategy is often the best.

In TD(lambda), **lambda** refers to the use of eligibility trace.

**Lambda return** is a weighted average of returns `G` which have different decay of lambda applied according to how many 
steps away to compute each return `G`

By eligibility trace, temporal difference methods will be efficient.

- `lambda = 1` Monte Carlo
- `lambda = 0` one-step TD

Eligibility trace provides the computational advantages to algorithmic mechanism.

Eligibility trace is a short-term memory vector `z_t` that parallels the long-term weight vector `w_t`.

`0 <= lambda <= 1` the **trace-decay parameter** determines the rate at which the trace falls.

Computational advantages of eligibility trace
- Only a single trace vector rather than the last n feature vectors
- Learning occurs continually and uniformly in time rather than being delayed or at the end of episodes
- Learning occurs and affect behavior immediately after a state

**Forward view** is what Monte Carlo and n-step TD are based on

**Backward view** uses the current TD error, looking backward to recently visited states using the **eligibility trace**.

**Compound update**, an update that averages simpler component updates

**Lambda-return**, the average is weighted proportionally to `lambda^(n - 1)`, a factor of `1 - lambda` is a normalization to 
make the weights sum to 1.

- When `lambda = 1`, updating according to lambda-return is a Monte Carlo algorithm.
- When `lambda = 0`, 

**Off-line lambda-return algorithm**, no changes to the weight vector during the episode, at the end of the episode, 
off-line updates are made.

## TD(Lambda)

- Update the weight vector on every step of an episode, so its estimates may be better sooner.
- Computations are equally distributed in time
- Can be applied to both episodic problems and continuing problems.

**Semi-gradient version of TD(Lambda) with function approximation

Eligibility trace vector assist in learning process and affects the **weight vector**.

Weight vector determines the estimated values.

In TD(Lambda), the weight vector is updated on each step proportional to the scalar TD error and the vector eligibility 
trace `w_t+1 = w_t + alpha delta_t z_t`. `delta_t` is TD error.

## Truncated TD(Lambda) (TTD(Lambda))

Lambda return algorithm, but lambda return needs to wait until the end of episode, so by using horizon `h`, truncate the 
lambda return summation, and replace the missing rewards in the future with estimated values.

## Online Lambda-return algorithm

**Online Lambda-return algorithm** is online, computing the weight vector `w_t` at each step `t` during an episode.

Drawback is that it's computationally complex.

## True online TD(Lambda)



Read from 12.4 redoing


