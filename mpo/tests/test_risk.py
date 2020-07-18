import numpy as np

# import pandas as pd
import cvxpy as cvx
from cvxpy.expressions.expression import Expression

# import mpo.mpo as M
import mpo.risk as risk
import mpo.simulation as sim
from mpo.common import KEY_STEP, KEY_WEIGHTS


def test_FactorRiskModel():
    # build risk model
    _, factor_cov, factor_exp, _, spec_cov = sim.generate_factor_returns(
        T=3, K=5, N=10
    )
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

    assert isinstance(model, risk.FactorRiskModel)

    w_next = cvx.Variable(10)
    kwargs = {KEY_WEIGHTS: w_next, KEY_STEP: 0}
    out = model.eval(**kwargs)
    print(f"out.shape: {out.shape}")
    assert isinstance(out, Expression)
    # assert output is scaler
    assert out.shape == ()
    assert out.is_dcp()


def test_RiskPenalty():
    # build risk model
    _, factor_cov, factor_exp, _, spec_cov = sim.generate_factor_returns(
        T=3, K=5, N=10
    )
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

    gamma = 0.1
    penalty = risk.RiskPenalty(gamma, model)

    w_next = cvx.Variable(10)
    # w_next = cvx.Constant(np.ones(10))
    kwargs = {KEY_WEIGHTS: w_next, KEY_STEP: 0}
    out = penalty.eval(**kwargs)

    print(out)

    assert isinstance(out, Expression)
    assert out.is_dcp()
