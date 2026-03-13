import numpy as np
import pandas as pd
from code.research_pipeline import ResearchPipeline
from code.data.provider import MarketDataProvider


class MockData(MarketDataProvider):

    def get_prices(self, tickers, start=None, end=None):
        days = 400
        data = {}
        for t in tickers:
            data[t] = np.cumprod(1 + np.random.normal(0, 0.01, days))
        return pd.DataFrame(data)

    def get_volumes(self, tickers):
        return pd.DataFrame(
            {t: np.random.uniform(1e6, 5e6, 300) for t in tickers}
        )

    def get_sectors(self, tickers):
        sectors = ["tech", "finance", "energy"]
        import random

        return {t: random.choice(sectors) for t in tickers}

    def get_market_caps(self, tickers):
        return {t: np.random.uniform(1e9, 1e12) for t in tickers}


data = MockData()

alpha_value = {
    "AAPL": 0.02,
    "MSFT": 0.015,
    "NVDA": -0.01,
    "TSLA": -0.02,
    "AMZN": 0.01,
}

alpha_momentum = {
    "AAPL": 0.01,
    "MSFT": 0.005,
    "NVDA": -0.005,
    "TSLA": -0.01,
    "AMZN": 0.003,
}

pipeline = ResearchPipeline(data)

weights, trades, cost = pipeline.build_portfolio(
    {"value": alpha_value, "momentum": alpha_momentum},
    current={"AAPL": 0.01},
    alpha_age=1,
)

print("Weights:", weights)
print("Trades:", trades)
print("Estimated cost:", cost)
