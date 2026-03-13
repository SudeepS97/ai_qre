class LiquidityModel:
    def __init__(self, data, adv_fraction=0.05):
        self.data = data
        self.adv_fraction = adv_fraction

    def max_position(self, tickers):
        vols = self.data.get_volumes(tickers)
        limits = {}
        for t in tickers:
            adv = vols[t].mean()
            limits[t] = adv * self.adv_fraction
        return limits
