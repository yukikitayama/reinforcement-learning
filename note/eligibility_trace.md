# Eligibility trace

In TD(lambda), **lambda** refers to the use of eligibility trace.

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

**Off-line lambda-return algorithm**, no changes to the weight vector during the episode, at the end of the episode, 
off-line updates are made.


