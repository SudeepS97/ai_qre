class ExecutionSimulator:

    def __init__(self, spread=0.0005, impact=0.1):
        self.spread = spread
        self.impact = impact

    def cost(self, trade):
        size = abs(trade)
        return size * self.spread + self.impact * (size**2)

    def portfolio_cost(self, trades):
        return sum(self.cost(t) for t in trades.values())
