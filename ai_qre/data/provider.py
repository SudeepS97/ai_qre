from datetime import datetime

import pandas as pd


class MarketDataProvider:
    def get_prices(
        self,
        tickers: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def get_returns(
        self,
        tickers: list[str],
        lookback: int = 252,
    ) -> pd.DataFrame:
        prices = self.get_prices(tickers)
        return prices.pct_change().dropna().tail(lookback)

    def get_volumes(self, tickers: list[str]) -> pd.DataFrame:
        raise NotImplementedError

    def get_sectors(self, tickers: list[str]) -> dict[str, str]:
        raise NotImplementedError

    def get_market_caps(self, tickers: list[str]) -> dict[str, float]:
        raise NotImplementedError
