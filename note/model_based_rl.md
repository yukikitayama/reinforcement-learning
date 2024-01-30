# Model-based reinforcement learning

A model of the environment is unknown in advance, but is learned from experience

The learned model is used to find a policy.

When the state and actions spaces are small enough, we find the following arrays
- `C[s, a]` The number of times action `a` has been chosen in state `s`
- `T_c[s, a, s']` The number of times this has resulted in a transition to state `s'`
- `R_s[s, a]` The sum of the resulting reward.

These are updated by the experience tuple `<s, a, r, s'>`
- `T_c[s, a, s'] := T_c[s, a, s'] + 1`
- `R_s[s, a] := R_s[s, a] + r`
- `C[s, a] := C[s, a] + 1`

And get the following estimates
- `T_tilda (s, a, s') = T_c[s, a, s'] / C[s, a]`
- `R_tilda (s, a) = R_s[s, a] / C[s, a]`

The value function is updated by `V[s] := max_a( R_tilda (s, a) + beta sum_s' T_tilda (s, a, s') V[s'] )`

