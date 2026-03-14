"""Tests for transaction cost in objective and turnover constraint."""

from collections.abc import Mapping, Sequence
from typing import cast

import cvxpy as cp
import numpy as np

from ai_qre.config import PortfolioConfig
from ai_qre.portfolio.constraints import basic_exposure_constraints
from ai_qre.portfolio.optimizer import PortfolioOptimizer
from ai_qre.types import CovarianceProvider, WeightVector


class _FixedCov:
    _matrix: np.ndarray

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = np.asarray(matrix, dtype=float)

    def compute(self, tickers: Sequence[str]) -> np.ndarray:
        return np.asarray(self._matrix, dtype=float) + 0.0


def test_turnover_limit_constraint_added() -> None:
    config = PortfolioConfig(
        net_target=0.0,
        gross_limit=2.0,
        turnover_limit=0.5,
    )
    tickers = ["A", "B"]
    weights_var = cp.Variable(2)
    current: Mapping[str, float] = {"A": 0.01, "B": -0.01}
    constraints = basic_exposure_constraints(
        config, tickers, weights_var, None, current=current
    )
    assert len(constraints) == 5


def test_optimizer_with_trading_cost_in_objective_returns_weights() -> None:
    tickers = ["A", "B"]
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    cov += 1e-6 * np.eye(2)
    alphas: WeightVector = {"A": 0.01, "B": 0.02}
    current: WeightVector = {"A": 0.0, "B": 0.0}
    config = PortfolioConfig(
        objective_type="mean_variance",
        use_trading_cost_in_objective=True,
        trading_cost_impact=0.05,
        risk_aversion=0.5,
        turnover_penalty=0.0,
    )
    opt = PortfolioOptimizer(cast(CovarianceProvider, _FixedCov(cov)), config)
    weights = opt.solve(alphas, current=current)
    assert set(weights.keys()) == set(tickers)
    assert abs(sum(weights.values()) - config.net_target) < 1e-4
