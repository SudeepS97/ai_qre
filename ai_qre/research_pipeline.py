from ai_qre.alpha.transforms import AlphaBlender, AlphaDecay, shrink
from ai_qre.capacity.liquidity import LiquidityModel
from ai_qre.config import CapacityConfig, PortfolioConfig, RiskConfig
from ai_qre.data.provider import MarketDataProvider
from ai_qre.execution.simulator import ExecutionSimulator
from ai_qre.portfolio.black_litterman import posterior_expected_returns
from ai_qre.portfolio.multi_period import solve_mpc_first_period
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

        tickers = list(alpha.keys())
        if (
            self.portfolio_config.use_black_litterman
            and self.portfolio_config.bl_views
            and tickers
        ):
            cov_matrix = self.cov.compute(tickers)
            alpha = posterior_expected_returns(
                tickers,
                alpha,
                cov_matrix,
                self.portfolio_config.bl_views,
                tau=float(self.portfolio_config.bl_tau),
                omega_scale=float(self.portfolio_config.bl_omega_scale),
            )

        optimizer = PortfolioOptimizer(self.cov, self.portfolio_config)
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

        trading_cost_lambda_diag = None
        if (
            self.portfolio_config.use_trading_cost_in_objective
            and tickers
            and current is not None
        ):
            trading_cost_lambda_diag = self.liquidity.trading_cost_impact_diag(
                tickers,
                float(self.portfolio_config.aum),
                base_impact=float(self.portfolio_config.trading_cost_impact),
            )

        weights = optimizer.solve(
            alpha,
            current=current,
            factor_exposures=factor_exposures,
            max_weight_by_asset=max_weight_by_asset,
            trading_cost_lambda_diag=trading_cost_lambda_diag,
        )
        current_weights = current or {}
        trades = {
            ticker: float(weights[ticker] - current_weights.get(ticker, 0.0))
            for ticker in weights
        }
        cost = self.exec_sim.portfolio_cost(trades)
        return weights, trades, cost

    def build_portfolio_mpc(
        self,
        alpha_models: AlphaModelMap,
        current: WeightVector | None = None,
        alpha_age: int | float = 0,
        use_factor_penalty: bool = True,
        mpc_horizon: int = 3,
        mpc_discount: float = 0.99,
    ) -> tuple[WeightVector, TradeVector, float]:
        """Build portfolio using MPC: solve multi-period problem, return first-period weights."""
        alpha = self.blender.blend(alpha_models)
        alpha = self.decay.apply(alpha, alpha_age)
        alpha = shrink(alpha)
        tickers = list(alpha.keys())
        if not tickers:
            return {}, {}, 0.0
        if (
            self.portfolio_config.use_black_litterman
            and self.portfolio_config.bl_views
        ):
            cov_matrix = self.cov.compute(tickers)
            alpha = posterior_expected_returns(
                tickers,
                alpha,
                cov_matrix,
                self.portfolio_config.bl_views,
                tau=float(self.portfolio_config.bl_tau),
                omega_scale=float(self.portfolio_config.bl_omega_scale),
            )
        cov_matrix = self.cov.compute(tickers)
        weights = solve_mpc_first_period(
            tickers,
            alpha,
            cov_matrix,
            current,
            self.portfolio_config,
            horizon=mpc_horizon,
            discount=mpc_discount,
        )
        current_weights = current or {}
        trades = {
            t: float(weights[t] - current_weights.get(t, 0.0)) for t in weights
        }
        cost = self.exec_sim.portfolio_cost(trades)
        return weights, trades, cost
