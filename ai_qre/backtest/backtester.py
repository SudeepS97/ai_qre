from collections.abc import Mapping

import pandas as pd


class Backtester:
    def run(
        self,
        weights: Mapping[str, float],
        returns: pd.DataFrame,
    ) -> pd.Series:
        weight_series = pd.Series(dict(weights), dtype=float)
        pnl = (returns * weight_series).sum(axis=1)
        one_plus_pnl: pd.Series = (pnl + 1.0).fillna(0.0)
        return one_plus_pnl.cumprod()
