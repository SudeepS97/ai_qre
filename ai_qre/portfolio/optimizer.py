from collections.abc import Mapping
from typing import cast

import cvxpy as cp
import pandas as pd

from ai_qre.config import PortfolioConfig
from ai_qre.portfolio.constraints import basic_exposure_constraints
from ai_qre.portfolio.objectives import (
    CvarInputs,
    CvarObjective,
    GlobalMinimumVarianceInputs,
    GlobalMinimumVarianceObjective,
    MeanVarianceInputs,
    MeanVarianceObjective,
    RobustMeanVarianceInputs,
    RobustMeanVarianceObjective,
    TrackingErrorInputs,
    TrackingErrorObjective,
)
from ai_qre.portfolio.resampling import resampled_efficiency_weights
from ai_qre.types import CovarianceProvider, WeightVector


class PortfolioOptimizer:
    def __init__(
        self,
        cov: CovarianceProvider,
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

        if self.config.use_resampled_efficiency:
            return resampled_efficiency_weights(
                self.cov,
                self.config,
                tickers,
                alphas,
                current=current,
            )

        cov_matrix = self.cov.compute(tickers)
        n_assets = len(tickers)
        weights_var = cp.Variable(n_assets)

        objective_terms: list[cp.Expression] = []

        objective_type = self.config.objective_type

        if objective_type == "gmv":
            gmv_inputs = GlobalMinimumVarianceInputs(cov_matrix=cov_matrix)
            gmv_objective = GlobalMinimumVarianceObjective(gmv_inputs)
            objective_terms.append(gmv_objective.build(tickers, weights_var))
        elif objective_type == "cvar":
            data = getattr(self.cov, "data", None)
            if data is None:
                raise ValueError(
                    "cvar objective requires a covariance provider with .data"
                )
            returns_df = data.get_returns(tickers)
            scenario_returns = returns_df.reindex(columns=tickers).to_numpy(
                dtype=float
            )
            cvar_inputs = CvarInputs(
                alphas=alphas,
                scenario_returns=scenario_returns,
                current=current,
                risk_aversion=float(self.config.risk_aversion),
                turnover_penalty=float(self.config.turnover_penalty),
            )
            cvar_objective = CvarObjective(cvar_inputs)
            objective_terms.append(cvar_objective.build(tickers, weights_var))
        elif objective_type == "tracking_error":
            benchmark_mapping: Mapping[str, float] = (
                self.config.benchmark_weights or {}
            )
            te_inputs = TrackingErrorInputs(
                alphas=alphas,
                cov_matrix=cov_matrix,
                current=current,
                benchmark=benchmark_mapping,
                risk_aversion=float(self.config.risk_aversion),
                turnover_penalty=float(self.config.turnover_penalty),
            )
            te_objective = TrackingErrorObjective(te_inputs)
            objective_terms.append(te_objective.build(tickers, weights_var))
        elif objective_type == "robust_mv":
            base_mv = MeanVarianceInputs(
                alphas=alphas,
                cov_matrix=cov_matrix,
                current=current,
                risk_aversion=float(self.config.risk_aversion),
                turnover_penalty=float(self.config.turnover_penalty),
            )
            robust_inputs = RobustMeanVarianceInputs(
                base=base_mv,
                uncertainty_radius=float(self.config.uncertainty_radius),
                uncertainty_type=self.config.uncertainty_type,
            )
            robust_objective = RobustMeanVarianceObjective(robust_inputs)
            objective_terms.append(
                robust_objective.build(tickers, weights_var)
            )
        else:
            mv_inputs = MeanVarianceInputs(
                alphas=alphas,
                cov_matrix=cov_matrix,
                current=current,
                risk_aversion=float(self.config.risk_aversion),
                turnover_penalty=float(self.config.turnover_penalty),
            )
            mv_objective = MeanVarianceObjective(mv_inputs)
            objective_terms.append(mv_objective.build(tickers, weights_var))

        max_weight_arg = None
        if max_weight_by_asset is not None:
            if isinstance(max_weight_by_asset, pd.Series):
                max_weight_arg = max_weight_by_asset
            else:
                max_weight_arg = dict(max_weight_by_asset)

        constraints: list[cp.Constraint] = basic_exposure_constraints(
            self.config, tickers, weights_var, max_weight_arg
        )

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
