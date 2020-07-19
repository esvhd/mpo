import numpy as np
import pandas as pd
from typing import Union
from abc import abstractmethod
import cvxpy as cvx
from cvxpy.expressions.expression import Expression

from mpo.common import (
    KEY_WEIGHTS,
    KEY_STEP,
    KEY_TRADE_WEIGHTS,
    KEY_BENCHMARK_WEIGHTS,
)

# KEY_WEIGHTS = "weights"
# KEY_TRADE_WEIGHTS = "trades"
# KEY_STEP = "timestep"


class BaseConstraint(object):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, **kwargs) -> Expression:
        pass


class MaxTurnover(BaseConstraint):
    def __init__(self, twoside_turnover):
        super().__init__()
        assert twoside_turnover > 0
        self.max_turnover = twoside_turnover

    def eval(self, **kwargs) -> Expression:
        trades = kwargs.get(KEY_TRADE_WEIGHTS)
        assert trades is not None
        return cvx.norm1(trades) <= self.max_turnover


class LongOnly(BaseConstraint):
    def __init__(self):
        super().__init__()

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        return weights >= 0


class MaxPositionLimit(BaseConstraint):
    def __init__(
        self,
        limits: Union[np.ndarray, pd.Series],
        bench_weights: Union[np.ndarray, pd.Series] = None,
    ):
        """Max position weight limits.

        Parameters
        ----------
        limits : Union[np.ndarray, pd.Series]
            Max weights. should be the same shape as weights.
            Cash can be modeled as a security, and therefore there can be a
            limit for cash.

            If limits.ndim > 1 then assume that these are indexed by time steps.
        bench_weights : Union[np.ndarray, pd.Series], optional
            Benchmark weights, by default None. If given then max weight limits
            are imposed on relative to benchmark weights.

            If bench_weights.ndim > 1 then assume that these are indexed by
            time steps.
        """
        super().__init__()
        # expects weights and limits to have the same shape
        assert isinstance(limits, np.ndarray) or isinstance(limits, pd.Series)
        self.limits = limits
        self.bench_weights = bench_weights

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        step = kwargs.get(KEY_STEP)

        if self.limits.ndim > 1:
            # different limits for different time steps
            limits = self.limits[step]
        else:
            limits = self.limits

        # make sure we have limits and weights match in dimension
        assert weights.shape == limits.shape

        if self.bench_weights is not None:
            # compute relative weights
            if self.bench_weights.ndim > 1:
                bench_wgts = self.bench_weights[step]
            else:
                bench_wgts = self.bench_weights

            assert weights.shape == bench_wgts.shape
            weights -= bench_wgts

        return weights <= limits


class MaxLeverageLimit(BaseConstraint):
    def __init__(self, max_leverage: float):
        super().__init__()
        self.max_leverage = max_leverage

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        # assume last position is CASH, so exclude
        return cvx.norm(weights[:-1], 1) <= self.max_leverage


class MinCash(BaseConstraint):
    def __init__(self, min_weight: float):
        super().__init__()
        self.min_weight = min_weight

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        # assume last position is CASH
        return weights[-1] >= self.min_weight


class NoBuy(BaseConstraint):
    def __init__(self, asset_idx: pd.Series):
        """No buy constraint. Currently only supports No Buy for same securities
        for all time steps. This can be expanded to support No Buy on different
        securities at different time steps.

        Parameters
        ----------
        asset_idx : pd.Series
            A list of security index number not to buy.
        """
        super().__init__()
        self.asset_idx = asset_idx

    def eval(self, **kwargs) -> Expression:
        trades = kwargs.get(KEY_TRADE_WEIGHTS)
        # asset list must be at most the same length as trade weight array
        assert trades.shape[0] >= len(self.asset_idx)
        return trades[self.asset_idx.values] <= 0


class NoSell(BaseConstraint):
    def __init__(self, asset_idx: pd.Series):
        super().__init__()
        self.asset_idx = asset_idx

    def eval(self, **kwargs) -> Expression:
        trades = kwargs.get(KEY_TRADE_WEIGHTS)
        # asset list must be at most the same length as trade weight array
        assert trades.shape[0] >= len(self.asset_idx)
        return trades[self.asset_idx.values] >= 0


class NoTrade(BaseConstraint):
    def __init__(self, asset_idx: pd.Series):
        super().__init__()
        self.asset_idx = asset_idx

    def eval(self, **kwargs) -> Expression:
        trades = kwargs.get(KEY_TRADE_WEIGHTS)
        # asset list must be at most the same length as trade weight array
        assert trades.shape[0] >= len(self.asset_idx)
        return trades[self.asset_idx.values] == 0.0


class BaseCost(object):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, **kwargs) -> Expression:
        pass


class TCost(BaseCost):
    def __init__(self, tcost_weights: Union[np.ndarray, pd.DataFrame]):
        """Transaction cost data. Here I leave tcost forecasting to a separate
        tasks, and we use the output directly here.

        Parameters
        ----------
        tcost_weights : Union[np.ndarray, pd.DataFrame]
            Tcost in % weight terms, for all time steps considered.
            This should match the time steps in return forecast.
            Cash position tcost should normally be 0.
        """
        super().__init__()
        self.tcost_weights = tcost_weights

    def eval(self, **kwargs):
        weights = kwargs.get(KEY_WEIGHTS)
        step = kwargs.get(KEY_STEP)

        tcost = self.tcost_weights
        if isinstance(tcost, pd.DataFrame):
            costs = tcost.iloc[step]
        else:
            # numpy
            costs = tcost[step]

        # total cost is measured in % weight
        total_cost = cvx.sum(cvx.multiply(weights, costs))
        return total_cost
