from collections.abc import Mapping

import numpy as np

from ai_qre.types import AlphaModelMap, AlphaVector


class AlphaBlender:
    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights: dict[str, float] = dict(weights or {})

    def blend(self, alpha_models: AlphaModelMap) -> AlphaVector:
        combined: AlphaVector = {}
        for name, alpha in alpha_models.items():
            model_weight = float(self.weights.get(name, 1.0))
            for ticker, value in alpha.items():
                combined[ticker] = float(
                    combined.get(ticker, 0.0) + model_weight * float(value)
                )
        return combined


class AlphaDecay:
    def __init__(self, half_life: float = 5.0) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.half_life = half_life

    def apply(self, alpha: AlphaVector, age_days: int | float) -> AlphaVector:
        factor = 0.5 ** (float(age_days) / self.half_life)
        return {
            ticker: float(value * factor) for ticker, value in alpha.items()
        }


def shrink(
    alpha: AlphaVector,
    prior_mean: float = 0.0,
    strength: float = 0.5,
) -> AlphaVector:
    clamped_strength = min(max(float(strength), 0.0), 1.0)
    return {
        ticker: float(
            (1.0 - clamped_strength) * value + clamped_strength * prior_mean
        )
        for ticker, value in alpha.items()
    }


def orthogonalize(alpha_matrix: np.ndarray) -> np.ndarray:
    if alpha_matrix.ndim != 2:
        raise ValueError("alpha_matrix must be 2-dimensional")
    q, _ = np.linalg.qr(alpha_matrix.T)
    return q.T
