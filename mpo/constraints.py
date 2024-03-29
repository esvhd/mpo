import numpy as np
import pandas as pd
from typing import Union, List
from abc import abstractmethod
import cvxpy as cvx
from cvxpy.expressions.expression import Expression

from mpo.common import (
    KEY_WEIGHTS,
    KEY_STEP,
    KEY_TRADE_WEIGHTS,
    KEY_BENCHMARK_WEIGHTS,
    KEY_ASSET_PROPERTY,
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
        """Max leverage limit.

        Parameters
        ----------
        max_leverage : float
            Max leverage, e.g. for 2x leverage set this to 2.0
        """
        super().__init__()
        self.max_leverage = max_leverage

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        # assume last position is CASH, so exclude
        # for norm calculation, see:
        # https://www.omnicalculator.com/math/matrix-norm
        # here we use infinite norm, which is the max of sums of all rows
        return cvx.norm(weights[:-1], p="inf") <= self.max_leverage


class MinCash(BaseConstraint):
    def __init__(self, min_weight: float):
        """Min cash pct constraint

        Parameters
        ----------
        min_weight : float
            min weight in decimal, e.g. for 1% set this to 0.01.
        """
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
            values of this series are indices of assets NOT to be bought.
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
        """[summary]

        Parameters
        ----------
        asset_idx : pd.Series
            values of this series are indices of assets NOT to be sold.
        """
        super().__init__()
        self.asset_idx = asset_idx

    def eval(self, **kwargs) -> Expression:
        trades = kwargs.get(KEY_TRADE_WEIGHTS)
        # asset list must be at most the same length as trade weight array
        assert trades.shape[0] >= len(self.asset_idx)
        return trades[self.asset_idx.values] >= 0


class NoTrade(BaseConstraint):
    def __init__(self, asset_idx: pd.Series):
        """No trade constraint

        Parameters
        ----------
        asset_idx : pd.Series
            values of this series are indices of assets NOT to be traded.
        """
        super().__init__()
        self.asset_idx = asset_idx

    def eval(self, **kwargs) -> Expression:
        trades = kwargs.get(KEY_TRADE_WEIGHTS)
        # asset list must be at most the same length as trade weight array
        assert trades.shape[0] >= len(self.asset_idx)
        return trades[self.asset_idx.values] == 0.0


class MaxAggregationConstraint(BaseConstraint):
    def __init__(
        self,
        agg_keys: pd.Series,
        key_limits: pd.Series,
        asset_property: pd.Series = None,
    ):
        """Max aggregation constraints, such as issuer, sector, rating, etc.

        N - no. of assets
        K - no. of unique aggregated groups

        Parameters
        ----------
        agg_keys : pd.Series
            Length N, the values of this series are the aggreation keys,
            e.g. ticker / sector / rating of each asset.
            This is converted into a one-hot matrix of shape (N, K)
        key_limits : pd.Series
            Length K, will be reorderd to match one-hot matrix column order.
        asset_prpoerty : pd.Series
            Length N, a property for assets, such as OAD for each bond,
            to be weighted in eval(),
            e.g. in sector or rating bucket constraints.
        """
        super().__init__()
        # make sure all agg keys have key limits
        keys = agg_keys.unique()
        flags = [x in key_limits.index for x in keys]
        # all keys have limits
        assert np.all(flags)
        # no. of keys match
        assert len(flags) == len(key_limits)

        # matrix (N, K), N - no. of assets, K - no. of unique agg keys
        self.agg_keys = pd.get_dummies(agg_keys)

        # make sure limits and aggregation matrices are aligned.
        # This is needed because later on, values and limits for each group
        # is compared in the correct order.
        assert len(key_limits) == len(self.agg_keys.columns)
        mask = [x in key_limits.index for x in self.agg_keys.columns]
        # all columns must be in key limit index
        assert np.all(mask)

        key_limits = key_limits.loc[self.agg_keys.columns]
        self.key_limits = key_limits

        # No. of assets match
        self.asset_property = asset_property
        if asset_property is not None:
            assert agg_keys.shape == asset_property.shape

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        assert weights is not None

        # weights must have the same length as agg_keys
        (N,) = weights.shape
        assert N == len(self.agg_keys)

        weights = cvx.reshape(weights, (1, N))
        assert weights.shape == (1, N)

        if self.asset_property is not None:
            # not a simple % weight constraint, more like CTSD.
            # reshape to (1, N)
            value = self.asset_property.values.reshape(weights.shape)
            weights = cvx.multiply(weights, value)
            assert weights.shape == (1, N)

        # (1, N) x (N, K) = (1, K)
        agg_vals = weights @ self.agg_keys.values
        _, K = agg_vals.shape
        vals = cvx.reshape(agg_vals, (K,))

        return vals <= self.key_limits.values


class AggregationEqualityConstraint(BaseConstraint):
    def __init__(
        self,
        asset_property: pd.DataFrame,
        target: Union[pd.Series, List[float]],
        tolerance=1e-7,
    ):
        """Full aggregation equality constraint, e.g. portfolio level equality
        constraint.

        Parameters
        ----------
        asset_property : pd.DataFrame
            asset properties, T x N, N assets, T time steps
        target: Union[pd.Series, List[float]]
            target value to be reached. length = T
        tolerance : [type], optional
            [description], by default 1e-5
        """
        super().__init__()

        assert asset_property.ndim == 2
        assert len(asset_property) == len(target)

        self.asset_property = asset_property
        self.tolerance = tolerance
        self.target = np.abs(target)

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        step = kwargs.get(KEY_STEP)
        assert weights is not None

        # weights must have the same length as no. of assets
        (N,) = weights.shape
        T, K = self.asset_property.shape
        assert N == K

        weights = cvx.reshape(weights, (1, N))
        assert weights.shape == (1, N)

        # props = cvx.reshape(self.asset_property.values, (N, -1))
        # find current time step, convert to N x 1
        props = self.asset_property.values[step].reshape((N, -1))
        # (1, N) x (N, 1) = (1, 1)
        agg_value = weights @ props

        assert agg_value.shape == (1, 1)
        # assert agg_value.shape == (1, T)

        # target = cvx.reshape(self.target, (1, T))
        target = self.target[step]

        # norm1 returns the max of each summed columns
        # diff = cvx.abs(cvx.norm1(agg_value - target))
        diff = cvx.abs(agg_value - target)
        diff_scaler = cvx.reshape(diff, ())

        # print(step, weights.shape, props.shape, agg_value.shape)
        # print("agg_value = \n", agg_value)
        # print(f"target value = {target}, diff = {diff_scaler}")

        # return np.all(diff <= self.tolerance)

        return diff_scaler <= self.tolerance


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
