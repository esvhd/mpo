import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple


def generate_factor_returns(T: int, K: int, N: int, seed=None) -> Tuple:
    """Generate normally distributed random returns for N periods of K factors.

    Parameters
    ----------
    T : int
        Time periods
    K : int
        no. of factors
    N : int
        no. of assets

    Returns
    -------
    Tuple
        r_factor : np.ndarray
            Factor returns, (T, K)
        F : np.ndarray
            Factor covariance, (K, K)
        X : np.ndarray
            Factor exposure, (N, K)
        V : np.ndarray
            Asset covariance, (N, N)
    """
    if seed is not None:
        np.random.seed(seed)

    # r_factor = factor return, divide by 100 to scale down
    r_factor = np.random.randn(T, K) / 10.0

    # F = factor covariance
    F = np.cov(r_factor.T)

    # check for positive definite
    assert np.all(np.linalg.eigvals(F) > 0), "Covariance not Positive-Definite."

    # generate factor exposure
    X = np.random.randn(N, K)

    # covariance matrix, assumptions is specific risk covariance = 0
    V = X @ F @ X.T

    specific_cov = np.zeros((N, N))

    return (r_factor, F, X, V, specific_cov)


def generate_forecasts(
    n_periods: int = 2, N: int = 10, seed=None
) -> pd.DataFrame:
    """Generate multi-period return forecasts.

    Parameters
    ----------
    n_periods : int, optional
        No. of periods / steps, by default 2
    N : int, optional
        No. of assets, by default 10

    Returns
    -------
    pd.Dataframe
        dataframe of return forecasts, shape (n_periods, N)
    """
    assert n_periods > 0
    assert N > 1
    periods = pd.date_range(
        start=dt.datetime.today().date(), periods=n_periods, freq="B"
    )

    if seed is not None:
        np.random.seed(seed)

    # random forcasts
    forecasts = pd.DataFrame(
        np.random.randn(n_periods, N) / 10.0, index=periods
    )

    return forecasts
