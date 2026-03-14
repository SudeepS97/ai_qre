"""CVXPY constraints for net/gross exposure, bounds, optional turnover limit."""

from collections.abc import Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from ai_qre.config import PortfolioConfig


def basic_exposure_constraints(
    config: PortfolioConfig,
    tickers: Sequence[str],
    weights_var: cp.Variable,
    max_weight_by_asset: pd.Series | dict[str, float] | None,
    current: Mapping[str, float] | None = None,
) -> list[cp.Constraint]:
    """Net target, gross limit, per-asset bounds, and optional turnover limit."""
    n_assets = len(tickers)
    upper_bounds = np.full(n_assets, float(config.max_position), dtype=float)
    if max_weight_by_asset is not None:
        per_asset_series = (
            max_weight_by_asset
            if isinstance(max_weight_by_asset, pd.Series)
            else pd.Series(dict(max_weight_by_asset), dtype=float)
        )
        per_asset = (
            per_asset_series.reindex(tickers)
            .fillna(float(config.max_position))
            .to_numpy(dtype=float)
        )
        upper_bounds = np.minimum(upper_bounds, per_asset)
    constraints: list[cp.Constraint] = [
        cp.sum(weights_var) == float(config.net_target),
        cp.norm1(weights_var) <= float(config.gross_limit),
        weights_var <= upper_bounds,
        weights_var >= -upper_bounds,
    ]
    if config.turnover_limit is not None and current is not None:
        current_array = np.asarray(
            [float(current.get(t, 0.0)) for t in tickers],
            dtype=float,
        )
        constraints.append(
            cp.norm1(weights_var - current_array)
            <= float(config.turnover_limit)
        )
    return constraints
