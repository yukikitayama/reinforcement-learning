# Math

## Proof of infinite sum of discounted rewards as finite number

1. Get a formula of the sum of the `n` finite geometric series
2. Take limit of `n -> inf`
3. `r` with the `n` exponent will vanish and get a finite number

- https://www.khanacademy.org/math/ap-calculus-bc/bc-series-new/bc-series-optional/v/infinite-geometric-series
- https://en.wikipedia.org/wiki/Geometric_series#Sum

Or

`sum gamma^t R`

Let `x = (gamma^0 + gamma^1 + gamma^2 + ...)`

`x = gamma^0 + gamma x` because `gamma x = gamma (gamma^0 + gamma^1 + gamma^2 + ...) = gamma^1 + gamma^2 + gamma^3 + ...`

`x = gamma^0 + gamma x`

`x - gamma x = gamma^0`

`x - gamma x = 1`

`x (1 - gamma) = 1`

`x = 1 / (1 - gamma)`

Then we multiply `R` and we get `R / (1 - gamma)`
