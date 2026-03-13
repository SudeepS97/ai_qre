from typing import TYPE_CHECKING

from ai_qre.alpha.transforms import AlphaBlender, AlphaDecay, shrink
from ai_qre.config import PortfolioConfig
from ai_qre.execution.simulator import ExecutionSimulator
from ai_qre.portfolio.optimizer import PortfolioOptimizer
from ai_qre.risk.covariance import ShrinkageCovariance

if TYPE_CHECKING:
    from ai_qre.data.provider import MarketDataProvider


class ResearchPipeline:
    def __init__(self, data: MarketDataProvider) -> None:
        self.data = data

        self.blender = AlphaBlender()
        self.decay = AlphaDecay()

        self.portfolio_config = PortfolioConfig()

        self.cov = ShrinkageCovariance(data)

        self.exec_sim = ExecutionSimulator()

    def build_portfolio(
        self,
        alpha_models: dict[str, dict[str, float]],
        current: dict[str, float] | None = None,
        alpha_age: int | float = 0,
    ) -> tuple[dict[str, float], dict[str, float], float]:
        alpha = self.blender.blend(alpha_models)
        alpha = self.decay.apply(alpha, alpha_age)
        alpha = shrink(alpha)

        optimizer = PortfolioOptimizer(self.cov, self.portfolio_config)

        weights = optimizer.solve(alpha, current)

        trades = {
            t: weights[t] - current.get(t, 0) if current else weights[t]
            for t in weights
        }

        cost = self.exec_sim.portfolio_cost(trades)

        return weights, trades, cost
