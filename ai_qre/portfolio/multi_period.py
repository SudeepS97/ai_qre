"""Multi-period / MPC-style portfolio optimization over a finite horizon."""

from collections.abc import Mapping, Sequence
from typing import cast

import cvxpy as cp
import numpy as np

from ai_qre.config import PortfolioConfig
from ai_qre.types import WeightVector


def solve_mpc_first_period(
    tickers: Sequence[str],
    alphas: Mapping[str, float],
    cov_matrix: np.ndarray,
    current: Mapping[str, float] | None,
    config: PortfolioConfig,
    horizon: int = 3,
    discount: float = 0.99,
) -> WeightVector:
    """Solve T-period problem with discounted utility; return first-period weights (MPC step)."""
    tickers_list = list(tickers)
    if not tickers_list or horizon < 1:
        return dict(alphas) if alphas else {}

    n = len(tickers_list)
    T = horizon
    current_mapping: Mapping[str, float] = current or {}
    w_prev = np.asarray(
        [float(current_mapping.get(t, 0.0)) for t in tickers_list],
        dtype=float,
    )
    alpha_vec = np.asarray(
        [float(alphas.get(t, 0.0)) for t in tickers_list],
        dtype=float,
    )
    Sigma = np.asarray(cov_matrix, dtype=float)
    if Sigma.shape != (n, n):
        Sigma = np.eye(n, dtype=float) * 0.01

    weight_vars = [cp.Variable(n) for _ in range(T)]
    terms: list[cp.Expression] = []
    constraints: list[cp.Constraint] = []
    net_target = float(config.net_target)
    gross_limit = float(config.gross_limit)
    max_pos = float(config.max_position)
    risk_aversion = float(config.risk_aversion)
    turnover_penalty = float(config.turnover_penalty)

    for t in range(T):
        w_t = weight_vars[t]
        w_prev_t = w_prev if t == 0 else weight_vars[t - 1]
        period_util = (
            alpha_vec @ w_t
            - risk_aversion * cp.quad_form(w_t, Sigma)
            - turnover_penalty * cp.norm1(w_t - w_prev_t)
        )
        terms.append(cast(cp.Expression, (discount**t) * period_util))
        constraints.extend(
            [
                cp.sum(w_t) == net_target,
                cp.norm1(w_t) <= gross_limit,
                w_t <= max_pos,
                w_t >= -max_pos,
            ]
        )

    problem = cp.Problem(cp.Maximize(cp.sum(terms)), constraints)
    problem.solve(solver=getattr(cp, config.solver, cp.OSQP))
    w0_val = weight_vars[0].value
    if w0_val is None:
        return {
            ticker: float(w_prev[i]) for i, ticker in enumerate(tickers_list)
        }
    return {ticker: float(w0_val[i]) for i, ticker in enumerate(tickers_list)}
