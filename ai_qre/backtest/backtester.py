import pandas as pd


class Backtester:
    def run(self, weights, returns):
        pnl = (returns * pd.Series(weights)).sum(axis=1)
        equity = (1 + pnl).cumprod()
        return equity
