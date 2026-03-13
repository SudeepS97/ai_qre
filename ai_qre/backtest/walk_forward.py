from dataclasses import dataclass

import pandas as pd

from ai_qre.config import WalkForwardConfig
from ai_qre.data.provider import MarketDataProvider
from ai_qre.types import AlphaGeneratorLike, ResearchPipelineLike, WeightVector


@dataclass(frozen=True)
class WalkForwardResult:
    equity_curve: pd.Series
    weights_by_rebalance: pd.DataFrame
    turnover_by_rebalance: pd.Series
    cost_by_rebalance: pd.Series


class WalkForwardBacktester:
    config: WalkForwardConfig

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def run(
        self,
        pipeline: ResearchPipelineLike,
        alpha_generator: AlphaGeneratorLike,
        data_provider: MarketDataProvider,
        tickers: list[str],
    ) -> WalkForwardResult:
        prices = data_provider.get_prices(tickers)
        returns = prices.pct_change().dropna().reindex(columns=tickers)
        cfg = self.config

        equity_parts: list[pd.Series] = []
        weights_records: dict[pd.Timestamp, WeightVector] = {}
        turnover_records: dict[pd.Timestamp, float] = {}
        cost_records: dict[pd.Timestamp, float] = {}
        current: WeightVector = {}
        running_nav = 1.0

        last_start = len(returns) - cfg.test_window
        for start in range(cfg.train_window, last_start + 1, cfg.step_size):
            train_slice = returns.iloc[start - cfg.train_window : start]
            test_slice = returns.iloc[start : start + cfg.test_window]
            if len(train_slice) < cfg.min_history or test_slice.empty:
                continue

            alpha_models = alpha_generator(train_slice)
            weights, trades, cost = pipeline.build_portfolio(
                alpha_models, current=current, alpha_age=0
            )
            rebalance_date = pd.Timestamp(test_slice.index[0])
            weights_records[rebalance_date] = weights
            turnover_records[rebalance_date] = float(
                sum(abs(value) for value in trades.values())
            )
            cost_records[rebalance_date] = float(cost)

            pnl = (
                (test_slice * pd.Series(weights, dtype=float))
                .sum(axis=1)
                .fillna(0.0)
            )
            first_index = pnl.index[0]
            pnl.loc[first_index] = float(pnl.loc[first_index] - cost)
            one_plus_pnl: pd.Series = (pnl + 1.0).fillna(0.0)
            segment: pd.Series = running_nav * one_plus_pnl.cumprod()
            running_nav = float(segment.iloc[-1])
            equity_parts.append(segment)
            current = weights

        if equity_parts:
            equity = pd.concat(equity_parts).sort_index()
            equity = equity[~equity.index.duplicated(keep="last")]
        else:
            equity = pd.Series(dtype=float, name="equity")

        weights_frame = pd.DataFrame.from_dict(
            weights_records, orient="index"
        ).sort_index()
        turnover_series = pd.Series(
            turnover_records, name="turnover", dtype=float
        ).sort_index()
        cost_series = pd.Series(
            cost_records, name="cost", dtype=float
        ).sort_index()
        return WalkForwardResult(
            equity_curve=equity,
            weights_by_rebalance=weights_frame,
            turnover_by_rebalance=turnover_series,
            cost_by_rebalance=cost_series,
        )
