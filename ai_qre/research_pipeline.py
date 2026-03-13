from ai_qre.alpha.transforms import AlphaBlender, AlphaDecay, shrink
from ai_qre.risk.covariance import ShrinkageCovariance
from ai_qre.portfolio.optimizer import PortfolioOptimizer
from ai_qre.execution.simulator import ExecutionSimulator
from ai_qre.config import PortfolioConfig


class ResearchPipeline:
    def __init__(self, data):
        self.data = data

        self.blender = AlphaBlender()
        self.decay = AlphaDecay()

        self.portfolio_config = PortfolioConfig()

        self.cov = ShrinkageCovariance(data)

        self.exec_sim = ExecutionSimulator()

    def build_portfolio(self, alpha_models, current=None, alpha_age=0):
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
