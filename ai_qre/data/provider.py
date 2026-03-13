from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    def get_prices(
        self,
        tickers: Sequence[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame: ...

    def get_returns(
        self,
        tickers: Sequence[str],
        lookback: int = 252,
    ) -> pd.DataFrame:
        prices = self.get_prices(tickers)
        return prices.pct_change().dropna().tail(lookback)

    def get_volumes(self, tickers: Sequence[str]) -> pd.DataFrame: ...

    def get_sectors(self, tickers: Sequence[str]) -> Mapping[str, str]: ...

    def get_market_caps(
        self, tickers: Sequence[str]
    ) -> Mapping[str, float]: ...
