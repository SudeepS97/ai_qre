from datetime import datetime

import numpy as np
import pandas as pd
from ai_qre.data.provider import MarketDataProvider
from ai_qre.research_pipeline import ResearchPipeline
from ai_qre.utils.logging import BoundLogger, get_logger

_LOGGER: BoundLogger = get_logger(__name__)


class MockData(MarketDataProvider):
    def get_prices(
        self,
        tickers: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        days = 400
        data: dict[str, np.ndarray] = {}
        for t in tickers:
            data[t] = np.cumprod(1 + np.random.normal(0, 0.01, days))
        return pd.DataFrame(data)

    def get_volumes(self, tickers: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {t: np.random.uniform(1e6, 5e6, 300) for t in tickers}
        )

    def get_sectors(self, tickers: list[str]) -> dict[str, str]:
        import random

        sectors = ["tech", "finance", "energy"]
        return {t: random.choice(sectors) for t in tickers}

    def get_market_caps(self, tickers: list[str]) -> dict[str, float]:
        return {t: float(np.random.uniform(1e9, 1e12)) for t in tickers}


data = MockData()

alpha_value: dict[str, float] = {
    "AAPL": 0.02,
    "MSFT": 0.015,
    "NVDA": -0.01,
    "TSLA": -0.02,
    "AMZN": 0.01,
}

alpha_momentum: dict[str, float] = {
    "AAPL": 0.01,
    "MSFT": 0.005,
    "NVDA": -0.005,
    "TSLA": -0.01,
    "AMZN": 0.003,
}

pipeline = ResearchPipeline(data)

weights: dict[str, float]
trades: dict[str, float]
cost: float
weights, trades, cost = pipeline.build_portfolio(
    {"value": alpha_value, "momentum": alpha_momentum},
    current={"AAPL": 0.01},
    alpha_age=1,
)

_LOGGER.info(
    "portfolio_built",
    weights=weights,
    trades=trades,
    estimated_cost=cost,
)
