class AlphaDecay:

    def __init__(self, half_life=5):
        self.half_life = half_life

    def apply(self, alpha, age_days):
        factor = 0.5 ** (age_days / self.half_life)
        return {k: v * factor for k, v in alpha.items()}
