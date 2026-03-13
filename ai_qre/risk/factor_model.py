from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_qre.data.provider import MarketDataProvider


class SimpleFactorModel:
    def __init__(self, data: MarketDataProvider) -> None:
        self.data = data

    def estimate(self, tickers: list[str]) -> dict[str, float]:
        r = self.data.get_returns(tickers)
        market = r.mean(axis=1)
        betas: dict[str, float] = {}
        for t in tickers:
            betas[t] = r[t].cov(market) / market.var()
        return betas
