"""Covariance estimation for the portfolio optimizer."""

from collections.abc import Sequence

import numpy as np

from ai_qre.data.provider import MarketDataProvider


class ShrinkageCovariance:
    """Sample covariance with diagonal shrinkage for stability."""

    def __init__(
        self,
        data: MarketDataProvider,
        shrinkage: float = 0.1,
    ) -> None:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between 0 and 1")
        self.data = data
        self.shrinkage = float(shrinkage)

    def compute(self, tickers: Sequence[str]) -> np.ndarray:
        """Return covariance matrix (square, same order as tickers) from historical returns."""
        tickers_list = list(tickers)
        if not tickers_list:
            return np.zeros((0, 0), dtype=float)
        returns = self.data.get_returns(tickers_list).reindex(
            columns=tickers_list
        )
        sample = returns.cov().to_numpy(dtype=float)
        diag = np.diag(np.diag(sample))
        combined = (1.0 - self.shrinkage) * sample + self.shrinkage * diag
        return np.asarray(combined, dtype=float)
