import numpy as np


class AlphaBlender:
    def __init__(self, weights=None):
        self.weights = weights or {}

    def blend(self, alpha_models):
        combined = {}
        for name, alpha in alpha_models.items():
            w = self.weights.get(name, 1.0)
            for t, v in alpha.items():
                combined[t] = combined.get(t, 0) + w * v
        return combined


class AlphaDecay:
    def __init__(self, half_life=5):
        self.half_life = half_life

    def apply(self, alpha, age_days):
        factor = 0.5 ** (age_days / self.half_life)
        return {k: v * factor for k, v in alpha.items()}


def shrink(alpha, prior_mean=0.0, strength=0.5):
    shrunk = {}
    for k, v in alpha.items():
        shrunk[k] = (1 - strength) * v + strength * prior_mean
    return shrunk


def orthogonalize(alpha_matrix):
    # alpha_matrix shape: signals x assets
    q, _ = np.linalg.qr(alpha_matrix.T)
    return q.T
