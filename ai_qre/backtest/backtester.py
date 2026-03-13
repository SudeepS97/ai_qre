import pandas as pd


class Backtester:
    def run(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
    ) -> pd.Series:
        pnl: pd.Series = (returns * pd.Series(weights)).sum(axis=1)
        equity: pd.Series = (1.0 + pnl).cumprod()
        return equity
