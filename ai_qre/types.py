from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, TypeAlias

import numpy as np
import pandas as pd

Ticker: TypeAlias = str
AlphaVector: TypeAlias = dict[Ticker, float]
AlphaModelMap: TypeAlias = dict[str, AlphaVector]
WeightVector: TypeAlias = dict[Ticker, float]
TradeVector: TypeAlias = dict[Ticker, float]
FactorExposureMap: TypeAlias = dict[pd.Timestamp, pd.DataFrame]


class CovarianceProvider(Protocol):
    def compute(self, tickers: Sequence[str]) -> np.ndarray: ...


class MarketDataProviderLike(Protocol):
    def get_prices(
        self,
        tickers: Sequence[Ticker],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame: ...

    def get_returns(
        self,
        tickers: Sequence[Ticker],
        lookback: int = 252,
    ) -> pd.DataFrame: ...

    def get_volumes(self, tickers: Sequence[Ticker]) -> pd.DataFrame: ...

    def get_sectors(
        self, tickers: Sequence[Ticker]
    ) -> Mapping[Ticker, str]: ...

    def get_market_caps(
        self, tickers: Sequence[Ticker]
    ) -> Mapping[Ticker, float]: ...


class ResearchPipelineLike(Protocol):
    def build_portfolio(
        self,
        alpha_models: AlphaModelMap,
        current: WeightVector | None = None,
        alpha_age: int | float = 0,
        use_factor_penalty: bool = True,
    ) -> tuple[WeightVector, TradeVector, float]: ...


class AlphaGeneratorLike(Protocol):
    def __call__(self, train_returns: pd.DataFrame) -> AlphaModelMap: ...
