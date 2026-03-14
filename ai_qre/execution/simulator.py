"""Execution cost model: spread plus quadratic impact."""

from collections.abc import Mapping


class ExecutionSimulator:
    """Per-trade cost = spread * size + impact * size^2; portfolio cost = sum over trades."""

    def __init__(
        self,
        spread: float = 0.0005,
        impact: float = 0.1,
    ) -> None:
        self.spread = float(spread)
        self.impact = float(impact)

    def cost(self, trade: float) -> float:
        """Cost for a single trade (signed); uses absolute size."""
        size = abs(float(trade))
        return float(size * self.spread + self.impact * (size**2))

    def portfolio_cost(self, trades: Mapping[str, float]) -> float:
        """Total cost across all trades (ticker -> signed trade)."""
        return float(sum(self.cost(trade) for trade in trades.values()))
