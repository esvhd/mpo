import numpy as np
import pandas as pd
import datetime as dt
from cvxpy.expressions.expression import Expression
import cvxpy as cvx
from typing import Iterable, Dict, Union


class MPOReturnForecast(object):
    def __init__(self, forecasts: pd.DataFrame):
        """Multi period forecasts

        Parameters
        ----------
        forecasts : pd.DataFrame
            Forecasts, shape (T, N), T - no. of time periods, N - no. of assets.
        """
        super().__init__()
        if isinstance(forecasts.index, pd.DatetimeIndex):
            forecasts = forecasts.sort_index()

        self.forecasts = forecasts

    def __str__(self):
        return self.forecasts.to_string()

    def num_periods(self) -> int:
        return self.forecasts.shape[0]

    def weighted_returns(self, weight: Expression, step: int) -> Expression:
        rtns = self.forecasts.iloc[[step]]
        # rtns.shape = (1, N)
        # expects weight.shape = (N, 1)
        sums = rtns.values @ weight
        # sums.shape = 1 x 1
        return sums

    def weighted_returns_multi(
        self,
        weights: Expression,
        start: Union[int, dt.datetime],
        end: Union[int, dt.datetime] = None,
    ) -> Expression:
        """Multi-step weighted sums

        Parameters
        ----------
        start : int
            start step index
        end : int
            end step index
        weights : Expression
            multi-step weights, shape (N, m), where m = start - end - 1

        Returns
        -------
        Expression
            Summed return of all steps.
        """
        use_dt = isinstance(start, dt.datetime)
        if end is None:
            if use_dt:
                end = start + dt.timedelta(days=1)
            else:
                end = start + 1
        if use_dt:
            rtns = self.forecasts.loc[start:end]
        else:
            rtns = self.forecasts.iloc[start:end]
        # rtns.shape == (m, N)
        sums = rtns.values @ weights
        return cvx.sum(sums)


class MPO(object):
    def __init__(
        self,
        forecasts: MPOReturnForecast,
        # trading_times,
        terminal_weights: pd.Series = None,
        costs: Iterable = None,
        constraints: Iterable = None,
        # lookahead_periods=None,
        solver=None,
        solver_opts: Dict = None,
    ):
        super().__init__()
        if not isinstance(forecasts, MPOReturnForecast):
            raise ValueError("forecasts must be of type MPOReturnForecast")
        self.forecasts = forecasts

        self.costs = []
        self.constraints = []

        if costs is not None:
            for cost in costs:
                self.costs.append(cost)

        if constraints is not None:
            for constraint in constraints:
                self.constraints.append(constraint)

        self.terminal_weights = terminal_weights
        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def solve(
        self, init_mkt_values: pd.Series, verbose: bool = True
    ) -> pd.Series:
        # normalise weights to 1
        nav = sum(init_mkt_values)
        assert nav > 0
        w = cvx.Constant(init_mkt_values.values / nav)
        if verbose:
            print(f"Initial weights: {w.value}")

        # solve for all periods
        sub_problems = []
        z_vars = []
        posn_wgts = []

        for tau in range(self.forecasts.num_periods()):
            z = cvx.Variable(w.shape)
            # next period weights = current weight + trade weights
            w_next = w + z

            obj = self.forecasts.weighted_returns_multi(w_next, start=tau)

            # trades must self fund
            con_exps = [cvx.Zero(cvx.sum(z))]

            # add other constraints
            if self.constraints is not None:
                kwargs = {"weights": w_next, "trades": z}
                cons = [con.eval(**kwargs) for con in self.constraints]
                con_exps += cons

            prob = cvx.Problem(cvx.Maximize(obj), constraints=con_exps)
            sub_problems.append(prob)
            z_vars.append(z)
            posn_wgts.append(w_next)
            w = w_next

            # print(f"z = {z}")
            # print(f"w_next = {w_next}")

        if self.terminal_weights is not None:
            sub_problems[-1].constraints += [
                w_next == self.terminal_weights.values
            ]

        obj_value = sum(sub_problems).solve(solver=self.solver, verbose=verbose)

        if verbose:
            print(f"Final objective value = {obj_value}")

            zd = {idx: z_vars[idx].value for idx in range(len(z_vars))}
            zd = pd.DataFrame(zd).T

            print("\nTrade weights:\n")
            print(zd.to_string(float_format="{:.1%}".format))

            zd = {idx: posn_wgts[idx].value for idx in range(len(posn_wgts))}
            zd = pd.DataFrame(zd).T
            print("\nPost-Trade Position weights:\n")
            print(zd.to_string(float_format="{:.1%}".format))
            # for idx in range(len(posn_wgts)):
            #     print(
            #         f"Step: {idx} - {posn_wgts[idx].value}, sum = {np.sum(posn_wgts[idx].value):.3e}"
            #     )
        trade_values = pd.Series(
            z_vars[0].value * nav, index=init_mkt_values.index
        )

        return trade_values
