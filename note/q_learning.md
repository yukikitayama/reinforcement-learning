# Q-learning

`Q[s, a] := (1 - alpha) Q[s, a] + alpha (r + beta max_a' Q[s', a'])`

Q-learning is a sampled, asynchronous method for estimating the optimal **state-action values (Q function)**.

Q function is useful, because if you have Q function, you can compute values and policy.
- `V(s) = max_a Q(s, a)` 
- `pi(s) = argmax_a Q(s, a)`

