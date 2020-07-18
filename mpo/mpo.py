import numpy as np
import pandas as pd
import datetime as dt
from cvxpy.expressions.expression import Expression
import cvxpy as cvx
from typing import Iterable, Dict, Union

import mpo.constraints as C
import mpo.risk as risk
from mpo.common import KEY_WEIGHTS, KEY_TRADE_WEIGHTS, KEY_STEP


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
        terminal_weights: pd.Series = None,
        costs: Iterable[C.BaseCost] = None,
        constraints: Iterable[C.BaseConstraint] = None,
        risk_penalty: risk.RiskPenalty = None,
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
        self.risk_penalty = risk_penalty
        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def solve(self, init_mkt_values: pd.Series, verbose: bool = True) -> Dict:
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

            kwargs = {KEY_WEIGHTS: w_next, KEY_TRADE_WEIGHTS: z, KEY_STEP: tau}

            obj = self.forecasts.weighted_returns_multi(w_next, start=tau)
            assert obj.is_dcp()

            # add tcosts:
            if self.costs is not None:
                # cost for all steps
                step_costs = [tc.eval(**kwargs) for tc in self.costs]

                if verbose:
                    print(f"Added {len(step_costs)} steps of T-costs.")

                # validation
                convex_flags = [tc.is_convex() for tc in step_costs]
                assert np.all(convex_flags)

                # add cost to objective function, i.e. reduce returns / rewards
                obj -= cvx.sum(step_costs)

            # trades must self fund
            con_exps = [cvx.Zero(cvx.sum(z))]

            # add other constraints
            if self.constraints is not None:
                cons = [con.eval(**kwargs) for con in self.constraints]
                con_exps += cons

            # validate that all constraints are DCP
            assert np.all((c.is_dcp() for c in con_exps))

            # add risk penalty term if given
            if self.risk_penalty is not None:
                if verbose:
                    print(f"Add risk penalty term.")
                p = self.risk_penalty.eval(**kwargs)
                assert p.is_dcp(), f"p.shape = {p.shape}, DCP = {p.is_dcp()}"
                obj -= p
                # obj -= self.risk_penalty.eval(**kwargs)

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

        trade_weights = {idx: z_vars[idx].value for idx in range(len(z_vars))}
        trade_weights = pd.DataFrame(trade_weights).T

        position_wgts = {
            idx: posn_wgts[idx].value for idx in range(len(posn_wgts))
        }
        position_wgts = pd.DataFrame(position_wgts).T

        if verbose:
            print(f"Final objective value = {obj_value}")

            trade_weights = {
                idx: z_vars[idx].value for idx in range(len(z_vars))
            }
            trade_weights = pd.DataFrame(trade_weights).T

            print("\nTrade weights:\n")
            print(trade_weights.to_string(float_format="{:.1%}".format))

            print("\nTurnover:\n")
            print(
                trade_weights.abs()
                .sum(axis=1)
                .to_string(float_format="{:.1%}".format)
            )

            print("\nPost-Trade Position weights:\n")
            print(position_wgts.to_string(float_format="{:.1%}".format))
            # for idx in range(len(posn_wgts)):
            #     print(
            #         f"Step: {idx} - {posn_wgts[idx].value}, sum = {np.sum(posn_wgts[idx].value):.3e}"
            #     )
        trade_values = pd.Series(
            z_vars[0].value * nav, index=init_mkt_values.index
        )

        output = dict()
        output["trade_values"] = trade_values
        output["objective"] = obj_value
        output["trade_weights"] = trade_weights
        output["position_weights"] = position_wgts

        return output
