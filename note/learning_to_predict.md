# Learning to predict by the methods of temporal differences

The paper is meant to prove the convergence and optimality of TD methods for important special cases.

Learning to predict == prediction learning,

Temporal-difference (TD) methods, incremental learning procedures for prediction problems.

TD methods are by the error (differences) between temporally successive predictions

TD methods are 
1. more incremental and easier to compute.
2. more efficient use of their experience.

As methods for efficiently learning to predict **arbitrary events**, not just goal-related ones.

The paper is concerned with **multi-step** prediction problems. In single-step problem, the temporal-difference methods 
cannot be distinguished from supervised-learning methods.

Notations
- `x_t` state
- `z` reward
- `w` modifiable parameters or weights for prediction
- `P_t` estimate of `z`, can be `P(x_t, w)`

**Widrow-Hoff rule (Delta rule, ADALINE, LMS filter)** linear learning methods to update weights `delta w_t = alpha (z - w^T x_t) x_t` (P.14).

**Backpropagation procedure (Generalized delta rule)** by Rumelhart is basically same as widrow-hoff rule except the model 
is replaced by a multi-layer network or a nonlinear function of `x` and `w`.

TD procedure replaces the error `z - P_t` (`z` is actual value and `P_t` is model prediction) with the sum of changes in predictions.

`z - P_t = sum from k=t to m (P_k+1 - P_k)` where `P_m+1 = z`

**TD(1) procedure** `delta w_t = alpha (P_t+1 - P_t) sum from k=1 to t delta_w P_k` is derived. This equation can be computed incrementally, 
and computationally better than supervised-learning methods. When `P_t` is a linear function, the equation is called 
**linear TD(1) procedure**.

**TD(lambda)** `delta w_t = alpha (P_t+1 - P_t) sum from k=1 to t lambda^t-k delta_w P_k` is a new procedure to introduce an exponential weighting with recency. TD(1) treats all the observations equally,
 but TD(lambda) puts different weights depending on how recent the predictions are.

The exponential form to `lambda` enables the incremental computation, because, letting `e_t` denote the sum, 
`e_t+1 = delta_w P_t+1 + lambda e_t`.

When we use `lambda = 0`, we get **TD(0)** `delta_w = alpha (P_t+1 - P_t) delta_w P_t`, and it's a procedure to use only the most recent observation. 
TD(0) is identical to supervised-learning except the actual outcome replaced with the next prediction. It means that 
**TD(0) and supervised-learning** uses the same learning mechanism but with the different errors.

Widrow-Hoff procedure only minimizes error **on the training set**, but it doesn't necessarily minimize error for future experience.

## My takeaway

- This paper describes the difference between the supervised-learning methods and temporal-difference methods
- This paper describes the computational advantages of the temporal-difference methods.


Read from 4. Theory of temporal-difference methods