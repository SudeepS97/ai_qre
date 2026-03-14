"""Alpha combination and transforms: blend, decay, shrink, orthogonalize."""

from collections.abc import Mapping

import numpy as np

from ai_qre.types import AlphaModelMap, AlphaVector


class AlphaBlender:
    """Combines multiple alpha models into one vector with optional per-model weights."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights: dict[str, float] = dict(weights or {})

    def blend(self, alpha_models: AlphaModelMap) -> AlphaVector:
        """Sum over models: for each ticker, sum (model_weight * score); default weight 1.0."""
        combined: AlphaVector = {}
        for name, alpha in alpha_models.items():
            model_weight = float(self.weights.get(name, 1.0))
            for ticker, value in alpha.items():
                combined[ticker] = float(
                    combined.get(ticker, 0.0) + model_weight * float(value)
                )
        return combined


class AlphaDecay:
    """Applies exponential decay to alpha by age (half-life in days)."""

    def __init__(self, half_life: float = 5.0) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.half_life = half_life

    def apply(self, alpha: AlphaVector, age_days: int | float) -> AlphaVector:
        """Scale each score by 0.5^(age_days/half_life)."""
        factor = 0.5 ** (float(age_days) / self.half_life)
        return {
            ticker: float(value * factor) for ticker, value in alpha.items()
        }


def shrink(
    alpha: AlphaVector,
    prior_mean: float = 0.0,
    strength: float = 0.5,
) -> AlphaVector:
    """Shrink each value toward prior_mean: (1-strength)*value + strength*prior_mean; strength in [0,1]."""
    clamped_strength = min(max(float(strength), 0.0), 1.0)
    return {
        ticker: float(
            (1.0 - clamped_strength) * value + clamped_strength * prior_mean
        )
        for ticker, value in alpha.items()
    }


def orthogonalize(alpha_matrix: np.ndarray) -> np.ndarray:
    """QR-based orthogonalization: rows = observations, columns = signals; returns orthogonalized matrix."""
    if alpha_matrix.ndim != 2:
        raise ValueError("alpha_matrix must be 2-dimensional")
    q, _ = np.linalg.qr(alpha_matrix.T)
    return q.T
