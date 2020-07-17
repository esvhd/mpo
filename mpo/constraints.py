import numpy as np
import pandas as pd
from typing import Union
from abc import abstractmethod
import cvxpy as cvx
from cvxpy.expressions.expression import Expression

from mpo.common import KEY_WEIGHTS, KEY_STEP

# KEY_WEIGHTS = "weights"
# KEY_TRADE_WEIGHTS = "trades"
# KEY_STEP = "timestep"


class BaseConstraint(object):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, **kwargs) -> Expression:
        pass


class LongOnly(BaseConstraint):
    def __init__(self):
        super().__init__()

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        return weights >= 0


class MaxPositionLimit(BaseConstraint):
    def __init__(self, limits: Union[np.ndarray, pd.Series]):
        """Max position weight limits.

        Parameters
        ----------
        limits : Union[np.ndarray, pd.Series]
            Max weights. should be the same shape as weights.
            Cash can be modeled as a security, and therefore there can be a
            limit for cash.
        """
        super().__init__()
        # expects weights and limits to have the same shape
        self.limits = limits

    def eval(self, **kwargs) -> Expression:
        weights = kwargs.get(KEY_WEIGHTS)
        return weights <= self.limits


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
