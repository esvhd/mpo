import numpy as np
import pandas as pd
import cvxpy as cvx
from cvxpy.expressions.expression import Expression
import mpo.mpo as M
import mpo.constraints as C
import mpo.simulation as sim
import mpo.risk as risk


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
    rtns = sim.generate_forecasts(n_periods=3, N=N, seed=9984)
    forecasts = M.MPOReturnForecast(rtns)
    exp_max = rtns.values[range(len(rtns)), rtns.idxmax(axis=1).values].sum()

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    cons = [C.LongOnly()]

    prob = M.MPO(forecasts, constraints=cons)
    output = prob.solve(init_mkt_values, verbose=True)

    trade_vals = output.get("trade_values")
    obj_val = output.get("objective")

    assert np.isclose(obj_val, exp_max, rtol=1.0e-7)
    assert len(trade_vals) == N
    assert not np.isnan(trade_vals).any()
    print(f"Trade values = {trade_vals}")


def test_mpo_tcost():
    # test MPO with no t-cost and constraints
    N = 10
    rtns = sim.generate_forecasts(n_periods=3, N=N, seed=9984)
    # simulate cash = 0 return
    rtns.values[:, -1] = 0
    forecasts = M.MPOReturnForecast(rtns)

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    # tcost, test constant tcost
    # stay in cash for the first trade
    tcost = np.ones_like(rtns) * 0.14
    # cash has no tcost
    tcost[:, -1] = 0.0
    costs = [C.TCost(tcost)]

    cons = [C.LongOnly()]

    prob = M.MPO(forecasts, constraints=cons, costs=costs)
    output = prob.solve(init_mkt_values, verbose=True)

    trade_vals = output.get("trade_values")
    obj_val = output.get("objective")

    assert len(trade_vals) == N
    assert not np.isnan(obj_val)
    assert not np.isnan(trade_vals).any()
    print(f"Trade values = {trade_vals}")

    assert np.isclose(obj_val, 0.14478828790994064, rtol=1.0e-7)


def test_factor_penalty():
    # build risk model
    _, factor_cov, factor_exp, _, spec_cov = sim.generate_factor_returns(
        T=3, K=5, N=10
    )
    # TODO build 3D risk models
    factor_cov_arr = np.tile(factor_cov.reshape((-1, 1)), 3).T.reshape(
        (3, 5, 5)
    )
    factor_exp_arr = np.tile(factor_exp.reshape((-1, 1)), 3).T.reshape(
        (3, 10, 5)
    )
    spec_cov_arr = np.tile(spec_cov.reshape((-1, 1)), 3).T.reshape((3, 10, 10))

    model = risk.FactorRiskModel(
        factor_exp_arr, factor_cov_arr, spec_cov_arr, has_benchmark=False
    )

    gamma = 1.0
    penalty = risk.RiskPenalty(gamma, model)

    # test MPO with no t-cost and constraints
    N = 10
    rtns = sim.generate_forecasts(n_periods=3, N=N, seed=9984)
    # simulate cash = 0 return
    rtns.values[:, -1] = 0
    forecasts = M.MPOReturnForecast(rtns)

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    # tcost, test constant tcost
    # stay in cash for the first trade
    tcost = np.ones_like(rtns) * 0.14
    # cash has no tcost
    tcost[:, -1] = 0.0
    costs = [C.TCost(tcost)]

    cons = [C.LongOnly()]

    # add risk model
    prob = M.MPO(forecasts, constraints=cons, costs=costs, risk_penalty=penalty)
    output = prob.solve(init_mkt_values, verbose=True)

    trade_vals = output.get("trade_values")
    obj_val = output.get("objective")

    assert len(trade_vals) == N
    assert not np.isnan(obj_val)
    assert not np.isnan(trade_vals).any()
    print(f"Trade values = {trade_vals}")

    # assert np.isclose(obj_val, 0.14478828790994064, rtol=1.0e-7)


def test_max_turnover():
    # test MPO with no t-cost and constraints
    N = 10
    rtns = sim.generate_forecasts(n_periods=3, N=N, seed=9984)
    forecasts = M.MPOReturnForecast(rtns)

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    MAX_TO = 0.5
    cons = [C.LongOnly(), C.MaxTurnover(MAX_TO)]

    prob = M.MPO(forecasts, constraints=cons)
    output = prob.solve(init_mkt_values, verbose=True)

    trade_vals = output.get("trade_values")
    trade_weights = output.get("trade_weights")

    assert len(trade_vals) == N
    assert not np.isnan(trade_vals).any()

    assert (trade_weights.abs().sum(axis=1) <= MAX_TO).all()
