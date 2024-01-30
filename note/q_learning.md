# Q-learning

`Q(S_t, A_t) <- Q(S_t, A_t) + alpha [ R_t+1 + gamma max_a Q(S_t+1, a) - Q(S_t, A_t) ]`

Or

`Q[s, a] := (1 - alpha) Q[s, a] + alpha (r + beta max_a' Q[s', a'])`

The learned action-value function `Q` directly approximates `q_*` the optimal action-value function.

Q-learning is **off-policy**

Q-learning is a sampled, asynchronous method for estimating the optimal **state-action values (Q function)**.

Q function is useful, because if you have Q function, you can compute values and policy.
- `V(s) = max_a Q(s, a)` 
- `pi(s) = argmax_a Q(s, a)`

