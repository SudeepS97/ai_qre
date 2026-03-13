class ExecutionSimulator:
    def __init__(
        self,
        spread: float = 0.0005,
        impact: float = 0.1,
    ) -> None:
        self.spread = spread
        self.impact = impact

    def cost(self, trade: float) -> float:
        size = abs(trade)
        return size * self.spread + self.impact * (size**2)

    def portfolio_cost(self, trades: dict[str, float]) -> float:
        return sum(self.cost(t) for t in trades.values())
