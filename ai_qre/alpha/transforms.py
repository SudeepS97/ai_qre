import numpy as np


class AlphaBlender:

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights: dict[str, float] = weights or {}

    def blend(
        self, alpha_models: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        combined: dict[str, float] = {}
        for name, alpha in alpha_models.items():
            w = self.weights.get(name, 1.0)
            for t, v in alpha.items():
                combined[t] = combined.get(t, 0) + w * v
        return combined


class AlphaDecay:
    def __init__(self, half_life: float = 5) -> None:
        self.half_life = half_life

    def apply(
        self, alpha: dict[str, float], age_days: int | float
    ) -> dict[str, float]:
        factor = 0.5 ** (age_days / self.half_life)
        return {k: v * factor for k, v in alpha.items()}


def shrink(
    alpha: dict[str, float],
    prior_mean: float = 0.0,
    strength: float = 0.5,
) -> dict[str, float]:
    shrunk: dict[str, float] = {}
    for k, v in alpha.items():
        shrunk[k] = (1 - strength) * v + strength * prior_mean
    return shrunk


def orthogonalize(alpha_matrix: np.ndarray) -> np.ndarray:
    # alpha_matrix shape: signals x assets
    q, _ = np.linalg.qr(alpha_matrix.T)
    return q.T
