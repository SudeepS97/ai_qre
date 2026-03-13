import random
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from ai_qre import ResearchPipeline, ResearchExtensions
from ai_qre.backtest.vectorized import VectorizedBacktestResult
from ai_qre.data.provider import MarketDataProvider
from ai_qre.tracking.experiment import ExperimentRun


class MockData(MarketDataProvider):
    def get_prices(
        self,
        tickers: Sequence[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        days = 500
        data: dict[str, np.ndarray] = {}
        for t in tickers:
            data[t] = 100 * np.cumprod(1 + np.random.normal(0, 0.01, days))
        return pd.DataFrame(
            data, index=pd.date_range("2020-01-01", periods=days, freq="B")
        )

    def get_returns(
        self, tickers: Sequence[str], lookback: int = 252
    ) -> pd.DataFrame:
        return self.get_prices(tickers).pct_change().dropna().tail(lookback)

    def get_volumes(self, tickers: Sequence[str]) -> pd.DataFrame:
        days = 300
        return pd.DataFrame(
            {t: np.random.uniform(5e5, 5e6, days) for t in tickers},
            index=pd.date_range("2021-01-01", periods=days, freq="B"),
        )

    def get_sectors(self, tickers: Sequence[str]) -> dict[str, str]:
        choices = ["tech", "finance", "industrial", "energy"]
        return {t: random.choice(choices) for t in tickers}

    def get_market_caps(self, tickers: Sequence[str]) -> dict[str, float]:
        return {t: float(np.random.uniform(2e9, 5e11)) for t in tickers}


tickers: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "XOM",
    "JPM",
    "BA",
]
data: MockData = MockData()
pipeline: ResearchPipeline = ResearchPipeline(data)
pipeline.portfolio_config.hard_factor_neutral = True
pipeline.portfolio_config.neutral_factors = ("market_beta", "size")
pipeline.portfolio_config.sector_neutral = True
pipeline.portfolio_config.use_capacity_limits = True
pipeline.portfolio_config.aum = 250_000_000.0

alpha_models: dict[str, dict[str, float]] = {
    "value": {t: float(np.random.normal(0, 0.02)) for t in tickers},
    "momentum": {t: float(np.random.normal(0, 0.02)) for t in tickers},
}
weights: dict[str, float]
trades: dict[str, float]
cost: float
weights, trades, cost = pipeline.build_portfolio(alpha_models)
print("weights", weights)
print("cost", cost)
print(
    pipeline.liquidity.capacity_report(
        weights, aum=pipeline.portfolio_config.aum
    ).head()
)

ext: ResearchExtensions = ResearchExtensions(data)
tracker: ExperimentRun = ext.experiments.start_run(
    "elite-demo", tags={"stage": "research"}
)
tracker.log_params(
    {"tickers": tickers, "portfolio_config": pipeline.portfolio_config}
)
tracker.log_metrics(
    {"cost": cost, "gross": sum(abs(v) for v in weights.values())}
)
tracker.log_artifact_json("weights.json", weights)
tracker.finalize()

returns_frame: pd.DataFrame = (
    data.get_prices(tickers).pct_change().dropna().tail(120)
)
vec: VectorizedBacktestResult = ext.vectorized.run(
    pd.DataFrame(
        np.random.normal(0, 1, (120, len(tickers))),
        index=pd.date_range("2022-01-01", periods=120, freq="B"),
        columns=tickers,
    ),
    returns_frame,
)
print(vec.equity_curve.tail())
