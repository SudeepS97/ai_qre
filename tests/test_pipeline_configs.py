"""Tests that ResearchPipeline runs with different objective types and configs."""

from collections.abc import Sequence
from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd

from ai_qre.backtest.portfolio_env import PortfolioEnv
from ai_qre.config import PortfolioConfig
from ai_qre.data.provider import MarketDataProvider
from ai_qre.portfolio.optimizer import PortfolioOptimizer
from ai_qre.research_pipeline import ResearchPipeline
from ai_qre.risk.covariance import ShrinkageCovariance
from ai_qre.types import CovarianceProvider, WeightVector


class _MockData(MarketDataProvider):
    def get_prices(
        self,
        tickers: Sequence[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        n = 252
        return pd.DataFrame(
            {
                t: 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
                for t in tickers
            },
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )

    def get_returns(
        self, tickers: Sequence[str], lookback: int = 252
    ) -> pd.DataFrame:
        return self.get_prices(tickers).pct_change().dropna().tail(lookback)

    def get_volumes(self, tickers: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {t: np.full(100, 1e6) for t in tickers},
            index=pd.date_range("2023-01-01", periods=100, freq="B"),
        )

    def get_sectors(self, tickers: Sequence[str]) -> dict[str, str]:
        return {t: "tech" for t in tickers}

    def get_market_caps(self, tickers: Sequence[str]) -> dict[str, float]:
        return {t: 1e10 for t in tickers}


def test_pipeline_gmv_returns_weights_sum_to_net_target() -> None:
    data = _MockData()
    pipeline = ResearchPipeline(data)
    pipeline.portfolio_config.objective_type = "gmv"
    pipeline.portfolio_config.net_target = 0.0
    tickers = ["A", "B", "C"]
    alpha_models = {"sig": {t: 0.01 for t in tickers}}
    weights, _, _ = pipeline.build_portfolio(alpha_models)
    assert set(weights.keys()) == set(tickers)
    assert abs(sum(weights.values()) - 0.0) < 1e-4


def test_pipeline_tracking_error_with_benchmark() -> None:
    data = _MockData()
    pipeline = ResearchPipeline(data)
    pipeline.portfolio_config.objective_type = "tracking_error"
    pipeline.portfolio_config.benchmark_weights = {
        "A": 0.33,
        "B": 0.33,
        "C": 0.34,
    }
    pipeline.portfolio_config.net_target = 0.0
    tickers = ["A", "B", "C"]
    alpha_models = {"sig": {t: 0.0 for t in tickers}}
    weights, _, _ = pipeline.build_portfolio(alpha_models)
    assert abs(sum(weights.values()) - 0.0) < 1e-4


def test_optimizer_robust_mv_returns_weights() -> None:
    tickers = ["A", "B"]
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float) + 1e-6 * np.eye(
        2
    )
    alphas: WeightVector = {"A": 0.01, "B": 0.02}

    class _FixedCov(ShrinkageCovariance):
        def compute(self, tickers_list):
            return cov + 0.0

    data = _MockData()
    cov_provider = _FixedCov(data, shrinkage=0.0)
    config = PortfolioConfig(
        objective_type="robust_mv",
        uncertainty_radius=0.01,
        uncertainty_type="box",
        net_target=0.0,
    )
    opt = PortfolioOptimizer(cast(CovarianceProvider, cov_provider), config)
    weights = opt.solve(alphas)
    assert set(weights.keys()) == set(tickers)


def test_portfolio_env_reset_step_loop() -> None:
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, (30, 2)),
        columns=["X", "Y"],
    )

    def cost_fn(trades: WeightVector) -> float:
        return 0.001 * sum(abs(v) for v in trades.values())

    env = PortfolioEnv(returns, ["X", "Y"], cost_fn, state_window=5)
    state, info = env.reset(seed=42)
    assert "positions" in state
    total_reward = 0.0
    for _ in range(3):
        action = {"X": 0.5, "Y": -0.5}
        state, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    assert "pnl" in info or "cost" in info
