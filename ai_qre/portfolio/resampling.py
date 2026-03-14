"""Resampled efficiency (Michaud) portfolio weights by averaging over simulated inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np

from ai_qre.config import PortfolioConfig
from ai_qre.types import CovarianceProvider, WeightVector


class _OptimizerInstanceLike(Protocol):
    """Protocol for optimizer instance: has solve(alphas, current=...)."""

    def solve(
        self,
        alphas: Mapping[str, float],
        current: Mapping[str, float] | None = None,
    ) -> WeightVector: ...


# Factory: (cov, config) -> optimizer instance. Use Callable to avoid Pyre "cannot instantiate protocol".
_OptimizerFactory = Callable[
    [CovarianceProvider, PortfolioConfig], _OptimizerInstanceLike
]


class _FixedCovariance:
    """Covariance provider that always returns a fixed matrix (for resampling)."""

    _matrix: np.ndarray

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = np.asarray(matrix, dtype=float)

    def compute(self, tickers: Sequence[str]) -> np.ndarray:
        return np.asarray(self._matrix, dtype=float) + 0.0


def resampled_efficiency_weights(
    cov: CovarianceProvider,
    config: PortfolioConfig,
    tickers: list[str],
    alphas: Mapping[str, float],
    optimizer_factory: _OptimizerFactory,
    current: Mapping[str, float] | None = None,
    n_simulations: int | None = None,
    seed: int | None = None,
) -> WeightVector:
    """Average portfolio weights over bootstrap resamples of (mu, Sigma). Caller passes optimizer_factory (e.g. PortfolioOptimizer class) to avoid circular import."""
    data = getattr(cov, "data", None)
    if data is None:
        raise ValueError(
            "resampled_efficiency_weights requires a covariance provider with .data"
        )
    n_sim = (
        n_simulations
        if n_simulations is not None
        else config.resampled_simulations
    )
    rng = np.random.default_rng(
        config.resampled_seed if seed is None else seed
    )
    returns_df = data.get_returns(tickers)
    returns_df = returns_df.reindex(columns=tickers).dropna(how="all")
    if returns_df.empty or len(returns_df) < 2:
        opt = optimizer_factory(cov, config)
        return opt.solve(alphas, current=current)
    returns = returns_df.to_numpy(dtype=float)
    T, n = returns.shape
    weight_list: list[dict[str, float]] = []
    base_config = PortfolioConfig(
        max_position=config.max_position,
        gross_limit=config.gross_limit,
        net_target=config.net_target,
        turnover_penalty=config.turnover_penalty,
        risk_aversion=config.risk_aversion,
        factor_penalty=config.factor_penalty,
        sector_neutral=config.sector_neutral,
        max_names=config.max_names,
        solver=config.solver,
        objective_type="mean_variance",
        hard_factor_neutral=config.hard_factor_neutral,
        neutral_factors=config.neutral_factors,
        factor_tolerance=config.factor_tolerance,
        use_capacity_limits=config.use_capacity_limits,
        aum=config.aum,
    )
    for _ in range(n_sim):
        idx = rng.integers(0, T, size=T)
        returns_m = returns[idx]
        mu_m = returns_m.mean(axis=0)
        centered = returns_m - mu_m
        Sigma_m = (centered.T @ centered) / float(T)
        if Sigma_m.shape != (n, n):
            Sigma_m = np.diag(np.clip(np.diag(Sigma_m), 1e-8, None))
        else:
            Sigma_m = (Sigma_m + Sigma_m.T) / 2.0
            Sigma_m += 1e-8 * np.eye(n)
        alphas_m = dict(zip(tickers, mu_m.tolist()))
        fixed_cov = _FixedCovariance(Sigma_m)
        opt = optimizer_factory(fixed_cov, base_config)
        w = opt.solve(alphas_m, current=current)
        weight_list.append(w)
    out: WeightVector = {}
    n_w = len(weight_list)
    for t in tickers:
        out[t] = sum(w.get(t, 0.0) for w in weight_list) / n_w
    return out
