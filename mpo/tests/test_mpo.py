import numpy as np
import pandas as pd
import cvxpy as cvx
from cvxpy.expressions.expression import Expression
import mpo.mpo as M
import mpo.constraints as C
import mpo.simulation as sim
import mpo.risk as risk


def leq(x, MAX_VAL, atol=1e-6):
    return np.less(x, MAX_VAL) | np.isclose(x, MAX_VAL, atol=atol)


def geq(x, MIN_VAL, atol=1e-6):
    return np.greater(x, MIN_VAL) | np.isclose(x, MIN_VAL, atol=atol)


def _run_mpo_simple(cons=None, costs=None, risk=None, seed=9984):
    # test MPO with no t-cost and constraints
    N = 10
    rtns = sim.generate_forecasts(n_periods=3, N=N, seed=seed)
    print(rtns.to_string(float_format="{:.1%}".format))
    forecasts = M.MPOReturnForecast(rtns)

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    prob = M.MPO(forecasts, constraints=cons, costs=costs, risk_penalty=risk)
    output = prob.solve(init_mkt_values, verbose=True)

    return output


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


def test_NoTrade():
    # test MPO with no t-cost and constraints
    N = 10
    rtns = sim.generate_forecasts(n_periods=3, N=N, seed=9984)
    print(rtns.to_string(float_format="{:.1%}".format))
    forecasts = M.MPOReturnForecast(rtns)

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    no_trade_idx = pd.Series([0, 7])
    cons = [C.LongOnly(), C.NoTrade(no_trade_idx)]

    prob = M.MPO(forecasts, constraints=cons)
    output = prob.solve(init_mkt_values, verbose=True)

    trade_vals = output.get("trade_values")
    trade_weights = output.get("trade_weights")

    assert len(trade_vals) == N
    assert not np.isnan(trade_vals).any()

    assert np.allclose(trade_vals[no_trade_idx.values], 0.0, atol=1e-6)
    assert np.allclose(
        trade_weights.iloc[:, no_trade_idx.values], 0.0, atol=1e-6
    )


def test_NoBuy():
    # test MPO with no t-cost and constraints
    N = 10
    rtns = sim.generate_forecasts(n_periods=3, N=N, seed=9984)
    print(rtns.to_string(float_format="{:.1%}".format))
    forecasts = M.MPOReturnForecast(rtns)

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    no_trade_idx = pd.Series([0, 7])
    cons = [C.LongOnly(), C.NoBuy(no_trade_idx)]

    prob = M.MPO(forecasts, constraints=cons)
    output = prob.solve(init_mkt_values, verbose=True)

    trade_vals = output.get("trade_values")
    trade_weights = output.get("trade_weights")

    assert len(trade_vals) == N
    assert not np.isnan(trade_vals).any()

    # robust way to test <= in numpy
    assert np.all(
        np.less(trade_vals[no_trade_idx.values], 0.0)
        | np.isclose(trade_vals[no_trade_idx.values], 0.0, atol=1e-6)
    )

    # assert np.allclose(trade_weights.iloc[:, no_trade_idx.values], 0.0)
    assert np.all(
        np.less(trade_weights.iloc[:, no_trade_idx.values], 0.0)
        | np.isclose(trade_weights.iloc[:, no_trade_idx.values], 0.0, atol=1e-6)
    )


def test_NoSell():
    # test MPO with no t-cost and constraints
    N = 10
    rtns = sim.generate_forecasts(n_periods=3, N=N, seed=9984)
    print(rtns.to_string(float_format="{:.1%}".format))
    forecasts = M.MPOReturnForecast(rtns)

    init_mkt_values = pd.Series([100.0 for _ in range(N)])

    no_trade_idx = pd.Series([0, 6])
    cons = [C.LongOnly(), C.NoSell(no_trade_idx)]

    prob = M.MPO(forecasts, constraints=cons)
    output = prob.solve(init_mkt_values, verbose=True)

    trade_vals = output.get("trade_values")
    trade_weights = output.get("trade_weights")

    assert len(trade_vals) == N
    assert not np.isnan(trade_vals).any()

    # robust way to test <= in numpy
    assert np.all(
        np.greater(trade_vals[no_trade_idx.values], 0.0)
        | np.isclose(trade_vals[no_trade_idx.values], 0.0, atol=1e-6)
    )

    # assert np.allclose(trade_weights.iloc[:, no_trade_idx.values], 0.0)
    assert np.all(
        np.greater(trade_weights.iloc[:, no_trade_idx.values], 0.0)
        | np.isclose(trade_weights.iloc[:, no_trade_idx.values], 0.0, atol=1e-6)
    )


def test_MaxPositionLimit_with_bench():
    # test max +10% vs bench
    MAX_VAL = 0.1
    max_limits = pd.Series(np.ones(10) * MAX_VAL)
    bench_wgts = pd.Series(np.ones(10) * 0.1)
    cons = [C.LongOnly(), C.MaxPositionLimit(max_limits, bench_wgts)]

    output = _run_mpo_simple(cons=cons)

    # trade_vals = output.get("trade_values")
    trade_weights = output.get("trade_weights")

    rel_wgt = trade_weights - bench_wgts

    # all relative weights should be <= 10%
    flags = np.less(rel_wgt, MAX_VAL) | np.isclose(rel_wgt, MAX_VAL, atol=1e-6)
    assert np.all(flags)

    # test multi-step limits
    max_limits = np.ones(10 * 3).reshape((3, 10)) * MAX_VAL
    bench_wgts = np.ones(10 * 3).reshape((3, 10)) * 0.1
    cons = [C.LongOnly(), C.MaxPositionLimit(max_limits, bench_wgts)]

    output = _run_mpo_simple(cons=cons)

    # trade_vals = output.get("trade_values")
    trade_weights = output.get("trade_weights")

    rel_wgt = trade_weights - bench_wgts

    # all relative weights should be <= 10%
    flags = np.less(rel_wgt, MAX_VAL) | np.isclose(rel_wgt, MAX_VAL, atol=1e-6)
    assert np.all(flags)


def test_MaxPositionLimit_no_bench():
    # test max +10% vs bench
    MAX_VAL = 0.2
    max_limits = pd.Series(np.ones(10) * MAX_VAL)
    cons = [C.LongOnly(), C.MaxPositionLimit(max_limits, bench_weights=None)]

    output = _run_mpo_simple(cons=cons)

    # trade_vals = output.get("trade_values")
    trade_weights = output.get("trade_weights")

    # all relative weights should be <= 10%
    flags = np.less(trade_weights, MAX_VAL) | np.isclose(
        trade_weights, MAX_VAL, atol=1e-6
    )

    assert np.all(flags)


def test_MaxAggConstraint():
    agg_keys = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    # cash limit can be 100%
    key_limits = pd.Series([0.2, 0.25, 0.3, 1.0])
    K = len(key_limits)

    # (N, K)
    one_hot = pd.get_dummies(agg_keys)
    assert one_hot.shape[1] == len(key_limits)
    key_limits = key_limits.loc[one_hot.columns]

    agg_con = C.MaxAggregationConstraint(agg_keys, key_limits)
    cons = [C.LongOnly(), agg_con]

    output = _run_mpo_simple(cons=cons)

    # dataframe of (T, N)
    positions = output.get("position_weights")

    agg_vals = positions.values @ one_hot.values
    agg_vals = pd.DataFrame(
        agg_vals, index=positions.index, columns=one_hot.columns
    )

    dist = key_limits.values.reshape(1, K) - agg_vals.values
    print(f"Dist to limits:\n\n {dist}")

    assert np.all(geq(dist, 0.0))


def test_MaxAggConstraint_with_property():
    agg_keys = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    asset_prop = pd.Series(np.ones_like(agg_keys.values) * 0.5)
    # cash limit can be 100%
    key_limits = pd.Series([0.2, 0.25, 0.3, 1.0]) * 0.5
    K = len(key_limits)

    # (N, K)
    one_hot = pd.get_dummies(agg_keys)
    assert one_hot.shape[1] == len(key_limits)
    key_limits = key_limits.loc[one_hot.columns]

    agg_con = C.MaxAggregationConstraint(agg_keys, key_limits, asset_prop)
    cons = [C.LongOnly(), agg_con]

    output = _run_mpo_simple(cons=cons)

    # dataframe of (T, N)
    positions = output.get("position_weights")
    T, _ = positions.shape

    # reshape property
    prop_mat = np.tile(asset_prop.values, T).reshape((T, -1))

    # [(T, N) \odot (T, N)] * (N, K)
    agg_vals = np.multiply(positions.values, prop_mat) @ one_hot.values
    agg_vals = pd.DataFrame(
        agg_vals, index=positions.index, columns=one_hot.columns
    )

    dist = key_limits.values.reshape(1, K) - agg_vals.values
    print(f"Dist to limits:\n\n {dist}")

    assert np.all(geq(dist, 0.0))
