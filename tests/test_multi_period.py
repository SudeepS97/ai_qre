"""Tests for multi-period / MPC optimizer."""

from collections.abc import Mapping

import numpy as np

from ai_qre.config import PortfolioConfig
from ai_qre.portfolio.multi_period import solve_mpc_first_period
from ai_qre.types import WeightVector


def test_solve_mpc_first_period_returns_weights() -> None:
    tickers = ["A", "B"]
    alphas: WeightVector = {"A": 0.01, "B": 0.02}
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    cov += 1e-6 * np.eye(2)
    current: WeightVector = {"A": 0.0, "B": 0.0}
    config = PortfolioConfig(
        net_target=0.0,
        gross_limit=2.0,
        risk_aversion=0.5,
        turnover_penalty=0.05,
    )
    weights = solve_mpc_first_period(
        tickers,
        alphas,
        cov,
        current,
        config,
        horizon=2,
        discount=0.99,
    )
    assert set(weights.keys()) == set(tickers)
    assert abs(sum(weights.values()) - config.net_target) < 1e-3


def test_solve_mpc_empty_tickers_returns_empty_or_prior() -> None:
    alphas: Mapping[str, float] = {}
    cov = np.eye(2, dtype=float)
    config = PortfolioConfig()
    out = solve_mpc_first_period([], alphas, cov, None, config, horizon=2)
    assert out == {}
