import numpy as np
import cvxpy as cvx
from cvxpy.expressions.expression import Expression
from abc import abstractmethod
from typing import Iterable, Union, List, Dict

from mpo.common import (
    KEY_WEIGHTS,
    # KEY_TRADE_WEIGHTS,
    KEY_STEP,
    KEY_BENCHMARK_WEIGHTS,
)


class BaseRiskModel(object):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, **kwargs) -> Expression:
        pass

    @abstractmethod
    def steps(self) -> int:
        pass


class FactorRiskModel(BaseRiskModel):
    def __init__(
        self,
        factor_exposure: np.ndarray,
        factor_cov: np.ndarray,
        specific_cov: np.ndarray,
        has_benchmark: bool = False,
    ):
        """Multi-step factor risk model

        Parameters
        ----------
        factor_exposure : np.ndarray
            Asset factor exposure matrices.
            Shape (t, N, K), t is time stamp, N - no. of assets, K - no. of
            factors.
        factor_cov : np.ndarray
            Factor covariance matrices.
            Shape (t, K, K)
        specific_cov : np.ndarray
            Specific risk covariance, shape (t, N, N)
        has_benchmark : bool, optional
            whether to compute portfolio or active factor risk, by default False
        """
        super().__init__()

        t1, N, K = factor_exposure.shape
        t2, k1, k2 = factor_cov.shape
        t3, n1, n2 = specific_cov.shape

        # all have the same time steps
        assert t1 == t2 == t3
        # assert no. of assets are the same
        assert N == n1 == n2
        # assert no. of factors are the same
        assert K == k1 == k2

        assert not np.isnan(factor_exposure).any()
        assert not np.isnan(factor_cov).any()
        assert not np.isnan(specific_cov).any()

        self.has_benchmark = has_benchmark
        self.factor_exposure = factor_exposure
        self.factor_cov = factor_cov
        self.specific_cov = specific_cov

        # pre-compute covariance matrix. easier for later
        self.cov = (
            # factor_exposure @ factor_cov @ factor_exposure.T + specific_cov
            factor_exposure @ factor_cov @ np.swapaxes(factor_exposure, 2, 1)
            + specific_cov
        )
        assert self.cov.shape == (t1, N, N)

    def eval(self, **kwargs) -> Expression:
        # determines which factor model to use
        step = kwargs.get(KEY_STEP)
        weights = kwargs.get(KEY_WEIGHTS)

        assert step is not None
        assert weights is not None

        # reshape for matrix algebra later
        (N,) = weights.shape
        weights = cvx.reshape(weights, (N, 1))
        assert N == self.factor_exposure.shape[1]

        if self.has_benchmark:
            bench_weights = kwargs.get(KEY_BENCHMARK_WEIGHTS)
            assert bench_weights is not None
            # TODO: cvxpy does not have a isnan() check, so not validating
            # nan benchmark weights here...
            try:
                check = np.isnan(bench_weights)
                assert not np.any(check)
            except TypeError:
                # cvxpy type, do nothing
                pass

            # assumes benchmark also has cash weights
            (N,) = bench_weights.shape
            bench_weights = cvx.reshape(bench_weights, (N, 1))
            # turn into active weights
            weights -= bench_weights

        # for notation, see my notes on optimization

        # shape = (K, N) x (N, 1) = (K, 1)
        # x_p = self.factor_cov.T @ weights
        # factor_risk = x_p.T @ self.factor_cov @ x_p
        # specific_risk = weights.T @ self.specific_cov @ weights
        # port_risk = factor_risk + specific_risk

        # (1, N) x (N, N) x (N, 1) = (1, 1)
        # port_risk = weights.T @ self.cov[step] @ weights
        cov = self.cov[step]
        # quad_form return weights.T @ cov @ weights
        port_risk = cvx.quad_form(weights, cov)

        # reshape to scaler
        risk = cvx.reshape(port_risk, ())
        return risk

    def steps(self) -> int:
        return self.factor_exposure.shape[0]


class RiskPenalty(object):
    def __init__(
        self, gamma: Union[float, Iterable[float]], risk_model: BaseRiskModel
    ):
        """Risk Penality term for optimization

        Parameters
        ----------
        gamma : Union[float, Iterable[float]]
            Risk penality coefficient, of lenth t. If float then use the same
            gamma for all steps.
        risk_model : BaseRiskModel
            [description]
        """
        super().__init__()
        self.risk_model = risk_model

        if isinstance(gamma, Iterable):
            assert len(gamma) == risk_model.steps()
            # self.gamma = cvx.Parameter(shape=(len(gamma),), value=gamma)
            self.gamma = gamma
        else:
            steps = risk_model.steps()
            self.gamma = np.ones(steps) * gamma
            # value = np.ones(steps) * gamma
            # self.gamma = cvx.Parameter(shape=(steps,), value=value)

    def eval(self, **kwargs) -> Expression:
        step = kwargs.get(KEY_STEP)
        risk = self.risk_model.eval(**kwargs)
        assert risk.is_dcp()
        gamma_t = self.gamma[step]
        return gamma_t * risk


class WorseCaseRisk(BaseRiskModel):
    def __init__(self, risk_models: List[BaseRiskModel], kwargs: Dict = None):
        super().__init__()

        self.risk_models = risk_models
        self.kwargs = kwargs

    def eval(self, **kwargs) -> Expression:
        risks = [
            model.eval(**kwargs, **self.kwargs) for model in self.risk_models
        ]
        return cvx.maximum(*risks)
