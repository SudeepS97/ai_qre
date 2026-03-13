from ai_qre.data.provider import MarketDataProvider


class LiquidityModel:
    def __init__(
        self,
        data: MarketDataProvider,
        adv_fraction: float = 0.05,
    ) -> None:
        self.data = data
        self.adv_fraction = adv_fraction

    def max_position(self, tickers: list[str]) -> dict[str, float]:
        vols = self.data.get_volumes(tickers)
        limits: dict[str, float] = {}
        for t in tickers:
            adv = vols[t].mean()
            limits[t] = adv * self.adv_fraction
        return limits
