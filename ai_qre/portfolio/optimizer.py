from collections.abc import Mapping
from typing import cast

import cvxpy as cp
import numpy as np
import pandas as pd

from ai_qre.config import PortfolioConfig
from ai_qre.risk.covariance import ShrinkageCovariance
from ai_qre.types import WeightVector


class PortfolioOptimizer:
    def __init__(
        self,
        cov: ShrinkageCovariance,
        config: PortfolioConfig,
    ) -> None:
        self.cov = cov
        self.config = config

    def solve(
        self,
        alphas: Mapping[str, float],
        current: Mapping[str, float] | None = None,
        factor_exposures: pd.DataFrame | None = None,
        max_weight_by_asset: pd.Series | Mapping[str, float] | None = None,
    ) -> WeightVector:
        tickers = list(alphas.keys())
        if not tickers:
            return {}

        alpha_vec = np.asarray(
            [float(alphas[ticker]) for ticker in tickers], dtype=float
        )
        cov_matrix = self.cov.compute(tickers)
        n_assets = len(tickers)
        weights_var = cp.Variable(n_assets)

        current_array = (
            np.asarray(
                [float(current.get(ticker, 0.0)) for ticker in tickers],
                dtype=float,
            )
            if current is not None
            else np.zeros(n_assets, dtype=float)
        )

        risk_term = cp.quad_form(weights_var, cov_matrix)
        turnover_term = cp.norm1(weights_var - current_array)
        objective_terms: list[cp.Expression] = [
            alpha_vec @ weights_var,
            -float(self.config.risk_aversion) * risk_term,
            -float(self.config.turnover_penalty) * turnover_term,
        ]

        upper_bounds = np.full(
            n_assets, float(self.config.max_position), dtype=float
        )
        if max_weight_by_asset is not None:
            per_asset_series = (
                max_weight_by_asset
                if isinstance(max_weight_by_asset, pd.Series)
                else pd.Series(dict(max_weight_by_asset), dtype=float)
            )
            per_asset = (
                per_asset_series.reindex(tickers)
                .fillna(float(self.config.max_position))
                .to_numpy(dtype=float)
            )
            upper_bounds = np.minimum(upper_bounds, per_asset)

        constraints: list[cp.Constraint] = [
            cp.sum(weights_var) == float(self.config.net_target),
            cp.norm1(weights_var) <= float(self.config.gross_limit),
            weights_var <= upper_bounds,
            weights_var >= -upper_bounds,
        ]

        if factor_exposures is not None and not factor_exposures.empty:
            aligned_exposures = (
                factor_exposures.reindex(index=tickers)
                .fillna(0.0)
                .astype(float)
            )
            exposure_matrix = aligned_exposures.to_numpy(dtype=float)
            factor_expression = exposure_matrix.T @ weights_var
            objective_terms.append(
                cast(
                    cp.Expression,
                    -float(self.config.factor_penalty)
                    * cp.sum_squares(factor_expression),
                )
            )
            hard_columns = self._hard_neutral_columns(aligned_exposures)
            if hard_columns:
                hard_matrix = aligned_exposures.loc[:, hard_columns].to_numpy(
                    dtype=float
                )
                hard_expression = hard_matrix.T @ weights_var
                tol = float(self.config.factor_tolerance)
                constraints.extend(
                    [hard_expression <= tol, hard_expression >= -tol]
                )

        objective = cp.Maximize(cp.sum(objective_terms))
        solver = getattr(cp, self.config.solver, cp.OSQP)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)

        value = weights_var.value
        if value is None:
            return {ticker: 0.0 for ticker in tickers}

        raw_weights: WeightVector = {
            ticker: float(value[index]) for index, ticker in enumerate(tickers)
        }
        return self._apply_max_names(raw_weights)

    def _hard_neutral_columns(self, exposures: pd.DataFrame) -> list[str]:
        hard_columns: list[str] = []
        if self.config.hard_factor_neutral:
            wanted = set(self.config.neutral_factors)
            hard_columns.extend(
                [column for column in exposures.columns if column in wanted]
            )
        if self.config.sector_neutral:
            hard_columns.extend(
                [
                    column
                    for column in exposures.columns
                    if column.startswith("sector_")
                ]
            )
        return list(dict.fromkeys(hard_columns))

    def _apply_max_names(self, weights: WeightVector) -> WeightVector:
        max_names = self.config.max_names
        if max_names is None or max_names >= len(weights):
            return weights
        ranked = sorted(
            weights.items(), key=lambda item: abs(item[1]), reverse=True
        )
        keep = {ticker for ticker, _ in ranked[:max_names]}
        return {
            ticker: (value if ticker in keep else 0.0)
            for ticker, value in weights.items()
        }
