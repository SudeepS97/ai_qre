from ai_qre.alpha.transforms import AlphaBlender, AlphaDecay, shrink
from ai_qre.capacity.liquidity import LiquidityModel
from ai_qre.config import CapacityConfig, PortfolioConfig, RiskConfig
from ai_qre.data.provider import MarketDataProvider
from ai_qre.execution.simulator import ExecutionSimulator
from ai_qre.portfolio.optimizer import PortfolioOptimizer
from ai_qre.risk.barra_model import BarraLikeRiskModel
from ai_qre.risk.covariance import ShrinkageCovariance
from ai_qre.types import AlphaModelMap, TradeVector, WeightVector


class ResearchPipeline:
    def __init__(self, data: MarketDataProvider) -> None:
        self.data = data
        self.blender = AlphaBlender()
        self.decay = AlphaDecay()
        self.portfolio_config = PortfolioConfig()
        self.risk_config = RiskConfig()
        self.capacity_config = CapacityConfig()
        self.cov = ShrinkageCovariance(
            data, shrinkage=self.risk_config.shrinkage
        )
        self.factor_risk = BarraLikeRiskModel(data, self.risk_config)
        self.liquidity = LiquidityModel(data, self.capacity_config)
        self.exec_sim = ExecutionSimulator()

    def build_portfolio(
        self,
        alpha_models: AlphaModelMap,
        current: WeightVector | None = None,
        alpha_age: int | float = 0,
        use_factor_penalty: bool = True,
    ) -> tuple[WeightVector, TradeVector, float]:
        alpha = self.blender.blend(alpha_models)
        alpha = self.decay.apply(alpha, alpha_age)
        alpha = shrink(alpha)

        optimizer = PortfolioOptimizer(self.cov, self.portfolio_config)
        tickers = list(alpha.keys())
        factor_exposures = (
            self.factor_risk.compute_factor_exposures(tickers)
            if use_factor_penalty and tickers
            else None
        )

        max_weight_by_asset = None
        if self.portfolio_config.use_capacity_limits and tickers:
            max_weight_by_asset = self.liquidity.max_weight_limits(
                tickers, aum=float(self.portfolio_config.aum)
            )

        weights = optimizer.solve(
            alpha,
            current=current,
            factor_exposures=factor_exposures,
            max_weight_by_asset=max_weight_by_asset,
        )
        current_weights = current or {}
        trades = {
            ticker: float(weights[ticker] - current_weights.get(ticker, 0.0))
            for ticker in weights
        }
        cost = self.exec_sim.portfolio_cost(trades)
        return weights, trades, cost
