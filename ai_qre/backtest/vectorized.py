"""Vectorized backtest: alpha panel -> rebalanced weights -> equity curve and turnover."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_qre.config import VectorizedResearchConfig
from ai_qre.types import FactorExposureMap


@dataclass(frozen=True)
class VectorizedBacktestResult:
    """Weights by rebalance date, daily portfolio returns, equity curve, turnover per rebalance."""

    weights: pd.DataFrame
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    turnover: pd.Series


class VectorizedResearchHarness:
    """Turns alpha time series into long/short weights (top_n/bottom_n or sign), optional factor neutralization, gross scaling."""

    config: VectorizedResearchConfig

    def __init__(self, config: VectorizedResearchConfig | None = None) -> None:
        self.config = config or VectorizedResearchConfig()

    def run(
        self,
        alpha_frame: pd.DataFrame,
        returns_frame: pd.DataFrame,
        factor_exposure_by_date: FactorExposureMap | None = None,
    ) -> VectorizedBacktestResult:
        """Rebalance every rebalance_frequency days; return weights, portfolio returns, equity curve, turnover."""
        alpha_sorted = alpha_frame.sort_index()
        returns_sorted = returns_frame.sort_index().reindex(
            columns=alpha_sorted.columns
        )
        rebalance_index = alpha_sorted.index[
            :: self.config.rebalance_frequency
        ]

        weight_records: dict[pd.Timestamp, pd.Series] = {}
        previous_weights = pd.Series(
            0.0, index=alpha_sorted.columns, dtype=float
        )
        turnover_by_date: dict[pd.Timestamp, float] = {}

        for rebalance_date in rebalance_index:
            scores = alpha_sorted.loc[rebalance_date].astype(float).fillna(0.0)
            weights = self._weights_from_scores(scores)
            if (
                self.config.neutralize_each_date
                and factor_exposure_by_date
                and rebalance_date in factor_exposure_by_date
            ):
                weights = self._neutralize(
                    weights, factor_exposure_by_date[rebalance_date]
                )
            weights = self._scale_to_gross(weights, self.config.gross)
            weight_records[pd.Timestamp(rebalance_date)] = weights
            turnover_by_date[pd.Timestamp(rebalance_date)] = float(
                (weights - previous_weights).abs().sum()
            )
            previous_weights = weights

        weights_frame = pd.DataFrame.from_dict(
            weight_records, orient="index"
        ).sort_index()
        daily_weights = (
            weights_frame.reindex(returns_sorted.index).ffill().fillna(0.0)
        )
        portfolio_returns = (
            (daily_weights * returns_sorted).sum(axis=1).fillna(0.0)
        )
        one_plus_returns: pd.Series = (portfolio_returns + 1.0).fillna(0.0)
        equity_curve = one_plus_returns.cumprod()
        turnover = pd.Series(
            turnover_by_date, name="turnover", dtype=float
        ).sort_index()
        return VectorizedBacktestResult(
            weights=weights_frame,
            portfolio_returns=portfolio_returns,
            equity_curve=equity_curve,
            turnover=turnover,
        )

    def _weights_from_scores(self, scores: pd.Series) -> pd.Series:
        cfg = self.config
        if cfg.top_n is not None:
            long_names = set(scores.nlargest(cfg.top_n).index)
        else:
            long_names = set(scores[scores > 0.0].index)

        short_names: set[str] = set()
        if cfg.long_short:
            if cfg.bottom_n is not None:
                short_names = set(scores.nsmallest(cfg.bottom_n).index)
            else:
                short_names = set(scores[scores < 0.0].index)

        weights = pd.Series(0.0, index=scores.index, dtype=float)
        if long_names:
            weights.loc[list(long_names)] = 1.0
        if short_names:
            weights.loc[list(short_names)] = -1.0
        return weights

    def _neutralize(
        self, weights: pd.Series, exposures: pd.DataFrame
    ) -> pd.Series:
        aligned_exposures = exposures.reindex(index=weights.index).fillna(0.0)
        exposure_matrix = aligned_exposures.to_numpy(dtype=float)
        if exposure_matrix.size == 0:
            return weights
        raw_weights = weights.to_numpy(dtype=float)
        gram = exposure_matrix.T @ exposure_matrix
        pinv_gram = np.linalg.pinv(gram)
        projection: np.ndarray = (
            exposure_matrix @ pinv_gram @ exposure_matrix.T @ raw_weights
        )
        neutral_weights = raw_weights - projection
        return pd.Series(neutral_weights, index=weights.index, dtype=float)

    @staticmethod
    def _scale_to_gross(weights: pd.Series, gross: float) -> pd.Series:
        current_gross = float(weights.abs().sum())
        if current_gross <= 0.0:
            return weights
        return weights * (float(gross) / current_gross)
