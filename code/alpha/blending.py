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
