import pandas as pd
import mpo.simulation as sim


def test_generate_factor_returns():
    T = 120
    K = 5
    N = 10
    fn, F, X, V = sim.generate_factor_returns(T, K, N)

    assert fn.shape == (T, K)
    assert F.shape == (K, K)
    assert X.shape == (N, K)
    assert V.shape == (N, N)


def test_generate_forecasts():
    n_periods = 3
    N = 10

    r = sim.generate_forecasts(n_periods, N)
    assert isinstance(r, pd.DataFrame)
    assert r.shape == (n_periods, N)
