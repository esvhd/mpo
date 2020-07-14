# import numpy as np
# import cvxpy as cvx
# from cvxpy.expressions.expression import Expression
from abc import abstractmethod


class BaseConstraint(object):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, **kwargs):
        pass


class LongOnly(BaseConstraint):
    def __init__(self, key: str = "weights"):
        super().__init__()
        self.key = key

    def eval(self, **kwargs):
        weights = kwargs.get(self.key)
        return weights >= 0
