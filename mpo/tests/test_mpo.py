import pandas as pd
import cvxpy as cvx
from cvxpy.expressions.expression import Expression
import mpo.mpo as M
import mpo.constraints as C
import mpo.simulation as sim


def test_MPOReturnForecast_weighted_returns():
    rtns = sim.generate_forecasts(n_periods=3, N=10)
    z = cvx.Variable(10)

    m = M.MPOReturnForecast(rtns)

    s = m.weighted_returns(weight=z, step=1)
    assert isinstance(s, Expression)
    assert s.shape == (1,)


def test_MPOReturnForecast_weighted_returns_multi():
    rtns = sim.generate_forecasts(n_periods=5, N=10)
    # 10 assets, 3 steps
    z = cvx.Variable((10, 3))

    m = M.MPOReturnForecast(rtns)

    s = m.weighted_returns_multi(weights=z, start=1, end=4)
    assert isinstance(s, Expression)
    assert s.shape == ()


def test_MPOReturnForecast_weighted_returns_multi_date():
    rtns = sim.generate_forecasts(n_periods=5, N=10)
    # 10 assets, 3 steps
    z = cvx.Variable((10, 3))

    m = M.MPOReturnForecast(rtns)
    start = rtns.index[0]
    end = rtns.index[-1]

    s = m.weighted_returns_multi(weights=z, start=start, end=end)
    assert isinstance(s, Expression)
    assert s.shape == ()


def test_mpo_simple():
    # test MPO with no t-cost and constraints
    N = 10
    rtns = sim.generate_forecasts(n_periods=3, N=N)
    forecasts = M.MPOReturnForecast(rtns)

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    cons = [C.LongOnly()]

    prob = M.MPO(forecasts, constraints=cons)
    trade_vals = prob.solve(init_mkt_values, verbose=True)

    assert len(trade_vals) == N
    print(f"Trade values = {trade_vals}")
