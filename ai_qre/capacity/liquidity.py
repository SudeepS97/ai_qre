from collections.abc import Mapping, Sequence

import pandas as pd

from ai_qre.config import CapacityConfig
from ai_qre.data.provider import MarketDataProvider


class LiquidityModel:
    data: MarketDataProvider
    config: CapacityConfig

    def __init__(
        self,
        data: MarketDataProvider,
        config: CapacityConfig | None = None,
    ) -> None:
        self.data = data
        self.config = config or CapacityConfig()

    def average_daily_dollar_volume(self, tickers: Sequence[str]) -> pd.Series:
        tickers_list = list(tickers)
        volumes = self.data.get_volumes(tickers_list).reindex(
            columns=tickers_list
        )
        prices = self.data.get_prices(tickers_list).reindex(
            columns=tickers_list
        )
        adv_shares = volumes.mean(axis=0)
        last_prices = prices.iloc[-1]
        addv = (adv_shares * last_prices).astype(float)
        addv.name = "addv"
        return addv

    def trading_cost_impact_diag(
        self,
        tickers: Sequence[str],
        aum: float,
        base_impact: float = 0.1,
    ) -> dict[str, float]:
        """Per-asset quadratic trading cost coefficient (higher for lower ADV)."""
        if aum <= 0.0:
            raise ValueError("aum must be positive")
        addv = self.average_daily_dollar_volume(tickers)
        out: dict[str, float] = {}
        for t in tickers:
            adv = float(addv.reindex([t]).fillna(0.0).iloc[0])
            denom = max(adv, 1.0)
            out[t] = base_impact * aum / denom
        return out

    def max_weight_limits(
        self, tickers: Sequence[str], aum: float
    ) -> pd.Series:
        if aum <= 0.0:
            raise ValueError("aum must be positive")
        cfg = self.config
        addv = self.average_daily_dollar_volume(tickers)
        liquidatable_dollars = (
            addv
            * float(cfg.participation_cap)
            * float(cfg.forecast_days_to_liquidate)
        )
        max_weight = (liquidatable_dollars / float(aum)).clip(
            lower=float(cfg.min_weight_cap)
        )
        max_weight.name = "max_weight"
        return max_weight.astype(float)

    def capacity_report(
        self, weights: Mapping[str, float], aum: float
    ) -> pd.DataFrame:
        tickers = list(weights.keys())
        max_weight = self.max_weight_limits(tickers, aum)
        actual_abs_weight = (
            pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0).abs()
        )
        safe_denom = max_weight.replace(0.0, pd.NA)
        capacity_usage = (actual_abs_weight / safe_denom).fillna(0.0)
        report = pd.concat(
            [
                actual_abs_weight.rename("abs_weight"),
                max_weight,
                capacity_usage.rename("capacity_usage"),
            ],
            axis=1,
        )
        return report.sort_values("capacity_usage", ascending=False)
