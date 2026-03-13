from collections.abc import Mapping


class ExecutionSimulator:
    def __init__(
        self,
        spread: float = 0.0005,
        impact: float = 0.1,
    ) -> None:
        self.spread = float(spread)
        self.impact = float(impact)

    def cost(self, trade: float) -> float:
        size = abs(float(trade))
        return float(size * self.spread + self.impact * (size**2))

    def portfolio_cost(self, trades: Mapping[str, float]) -> float:
        return float(sum(self.cost(trade) for trade in trades.values()))
