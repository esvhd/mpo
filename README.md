# Multi-Period Portfolio Optimisation

My humble attempt to re-write [`cvxportfolio` package](https://github.com/cvxgrp/cvxportfolio).

I did this mainly because it's easier for me to have the inputs in the format
I like. Plus it's a good way to make sure my understanding was correct.

Happy optimizing. Contributon welcome.

## Data Dimensions

See code docs for exact data types for these inputs.

* `T` - timestep index
* `N` - no. of assets, including cash as there are many cash related
constraints. Cash item is assumed to be the **last** item, i.e. $N^{th}$ item.
* `K` - no. of factors in a factor model

**Return forecasts** `pd.dataframe` of shape: `(T, N)` .

To use **factor risk model**, multiple factor related matrices must be provided,
such as factor and specific risk covariance, factor exposure.

* Factor Covariance `np.ndarray` of shape: `(T, K, K)`
* Specific Covariance `np.ndarray` ofshape: `(T, N, N)`
* Factor Exposure `np.ndarray` ofshape: `(T, N, K)`

**Transaction Costs** can be either:s

* `np.ndarray` of shape `(N, )`, or
* `pd.dataframe` of shape `(T, N)`
