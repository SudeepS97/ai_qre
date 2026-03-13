class MarketDataProvider:

    def get_prices(self, tickers, start=None, end=None):
        raise NotImplementedError

    def get_returns(self, tickers, lookback=252):
        prices = self.get_prices(tickers)
        return prices.pct_change().dropna().tail(lookback)

    def get_volumes(self, tickers):
        raise NotImplementedError

    def get_sectors(self, tickers):
        raise NotImplementedError

    def get_market_caps(self, tickers):
        raise NotImplementedError
