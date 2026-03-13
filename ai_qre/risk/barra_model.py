from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_qre.config import RiskConfig
from ai_qre.data.provider import MarketDataProvider


@dataclass(frozen=True)
class FactorRiskSnapshot:
    exposures: pd.DataFrame
    factor_cov: pd.DataFrame
    idiosyncratic_var: pd.Series
    asset_cov: pd.DataFrame


class BarraLikeRiskModel:
    """A lightweight, production-style factor risk model.

    Factors:
      - market_beta
      - size (z-scored log market cap)
      - momentum (z-scored trailing return)
      - sector dummies
    """

    data: MarketDataProvider
    config: RiskConfig

    def __init__(
        self,
        data: MarketDataProvider,
        config: RiskConfig | None = None,
    ) -> None:
        self.data = data
        self.config = config or RiskConfig()

    def compute_factor_exposures(self, tickers: Sequence[str]) -> pd.DataFrame:
        tickers_list = list(tickers)
        returns = self.data.get_returns(
            tickers_list, lookback=self.config.factor_window
        ).reindex(columns=tickers_list)
        prices = self.data.get_prices(tickers_list).reindex(
            columns=tickers_list
        )
        market = returns.mean(axis=1)

        caps = (
            pd.Series(self.data.get_market_caps(tickers_list), dtype=float)
            .reindex(tickers_list)
            .fillna(1.0)
        )
        size_raw = pd.Series(
            np.log(caps.clip(lower=1.0).to_numpy(dtype=float)),
            index=caps.index,
        )
        size = self._zscore(size_raw)

        momentum_raw = (
            prices.pct_change(self.config.momentum_lookback)
            .iloc[-1]
            .reindex(tickers_list)
            .fillna(0.0)
        )
        momentum = self._zscore(momentum_raw)

        market_var = float(market.var())
        beta_values: dict[str, float] = {}
        for ticker in tickers_list:
            beta_values[ticker] = (
                float(returns[ticker].cov(market) / market_var)
                if market_var > 0.0
                else 0.0
            )
        beta = (
            pd.Series(beta_values, dtype=float)
            .reindex(tickers_list)
            .fillna(0.0)
        )

        sectors = (
            pd.Series(self.data.get_sectors(tickers_list), dtype="object")
            .reindex(tickers_list)
            .fillna("unknown")
        )
        sector_dummies = pd.get_dummies(sectors, prefix="sector", dtype=float)

        exposures = pd.DataFrame(
            {
                "market_beta": beta,
                "size": size,
                "momentum": momentum,
            },
            index=tickers_list,
        ).fillna(0.0)
        exposures = pd.concat([exposures, sector_dummies], axis=1).fillna(0.0)
        return exposures.astype(float)

    def factor_covariance(self, tickers: Sequence[str]) -> pd.DataFrame:
        tickers_list = list(tickers)
        exposures = self.compute_factor_exposures(tickers_list)
        returns = self.data.get_returns(
            tickers_list, lookback=self.config.factor_window
        ).reindex(columns=tickers_list)
        x = exposures.to_numpy(dtype=float)
        x_pinv = np.linalg.pinv(x)

        factor_returns: list[np.ndarray] = []
        for _, row in returns.iterrows():
            asset_returns = (
                row.reindex(tickers_list).fillna(0.0).to_numpy(dtype=float)
            )
            factor_returns.append(x_pinv @ asset_returns)

        factor_return_frame = pd.DataFrame(
            factor_returns, columns=exposures.columns, index=returns.index
        )
        return factor_return_frame.cov().astype(float)

    def snapshot(self, tickers: Sequence[str]) -> FactorRiskSnapshot:
        tickers_list = list(tickers)
        exposures = self.compute_factor_exposures(tickers_list)
        factor_cov = self.factor_covariance(tickers_list)
        returns = self.data.get_returns(
            tickers_list, lookback=self.config.factor_window
        ).reindex(columns=tickers_list)

        x = exposures.to_numpy(dtype=float)
        factor_cov_array = factor_cov.to_numpy(dtype=float)
        factor_component: np.ndarray = x @ factor_cov_array @ x.T
        sample_cov = returns.cov().to_numpy(dtype=float)
        residual_diag = np.clip(
            np.diag(sample_cov - factor_component), a_min=1e-8, a_max=None
        )
        asset_cov = factor_component + np.diag(residual_diag)

        return FactorRiskSnapshot(
            exposures=exposures,
            factor_cov=factor_cov,
            idiosyncratic_var=pd.Series(
                residual_diag, index=tickers_list, name="idio_var", dtype=float
            ),
            asset_cov=pd.DataFrame(
                asset_cov,
                index=tickers_list,
                columns=tickers_list,
                dtype=float,
            ),
        )

    @staticmethod
    def _zscore(values: pd.Series) -> pd.Series:
        std = float(values.std(ddof=0))
        if std <= 0.0:
            return pd.Series(0.0, index=values.index, dtype=float)
        return ((values - float(values.mean())) / std).astype(float)
