"""Simple single-factor (market) risk model."""

from ai_qre.data.provider import MarketDataProvider


class SimpleFactorModel:
    """Market beta only: regression of asset returns on equal-weight market return."""

    def __init__(self, data: MarketDataProvider) -> None:
        self.data = data

    def estimate(self, tickers: list[str]) -> dict[str, float]:
        """Return ticker -> market beta from historical returns."""
        r = self.data.get_returns(tickers)
        market = r.mean(axis=1)
        betas: dict[str, float] = {}
        for t in tickers:
            betas[t] = r[t].cov(market) / market.var()
        return betas
