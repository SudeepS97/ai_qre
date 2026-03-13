from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ai_qre.data.provider import MarketDataProvider


class ShrinkageCovariance:
    def __init__(
        self,
        data: MarketDataProvider,
        shrinkage: float = 0.1,
    ) -> None:
        self.data = data
        self.shrinkage = shrinkage

    def compute(self, tickers: list[str]) -> np.ndarray:
        r = self.data.get_returns(tickers)
        sample = r.cov().values
        diag = np.diag(np.diag(sample))
        combined = (1.0 - self.shrinkage) * sample + self.shrinkage * diag
        return np.asarray(combined)
