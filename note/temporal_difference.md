# Temporal difference

More supervised

- Model-based
- Value-function-based (Model free)
- Policy search

More direct learning

Many researches are focused on value-function-based

`v_t(s) = v_t-1(s) + alpha_t (r_t(s) - v_t-1(s))`

and `lim t->inf v_t(s) = v(s)` meaning `v_t(s)` will be actual average of returns when `t` is big enough

## Learning rate

- Learning rate has to be big enough, so that you can move to what the true value is, no matter where you start
- But the learning rate can't be so big that they don't damp out the noise and actually do a proper job of averaging.