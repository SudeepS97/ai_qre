import random
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from ai_qre import ResearchPipeline, ResearchExtensions
from ai_qre.backtest.vectorized import VectorizedBacktestResult
from ai_qre.data.provider import MarketDataProvider
from ai_qre.tracking.experiment import ExperimentRun
from ai_qre.utils.logging import (
    BoundLogger,
    get_logger,
    init_structured_logging,
)
from ai_qre.utils.position_sizing import (
    shares_to_long_short,
    weights_to_shares,
)

logger: BoundLogger = get_logger(__name__)


class MockData(MarketDataProvider):
    def get_prices(
        self,
        tickers: Sequence[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        days = 500
        data: dict[str, np.ndarray] = {}
        for t in tickers:
            data[t] = 100 * np.cumprod(1 + np.random.normal(0, 0.01, days))
        return pd.DataFrame(
            data, index=pd.date_range("2020-01-01", periods=days, freq="B")
        )

    def get_returns(
        self, tickers: Sequence[str], lookback: int = 252
    ) -> pd.DataFrame:
        return self.get_prices(tickers).pct_change().dropna().tail(lookback)

    def get_volumes(self, tickers: Sequence[str]) -> pd.DataFrame:
        days = 300
        return pd.DataFrame(
            {t: np.random.uniform(5e5, 5e6, days) for t in tickers},
            index=pd.date_range("2021-01-01", periods=days, freq="B"),
        )

    def get_sectors(self, tickers: Sequence[str]) -> dict[str, str]:
        choices = ["tech", "finance", "industrial", "energy"]
        return {t: random.choice(choices) for t in tickers}

    def get_market_caps(self, tickers: Sequence[str]) -> dict[str, float]:
        return {t: float(np.random.uniform(2e9, 5e11)) for t in tickers}


def main() -> None:
    init_structured_logging()
    tickers: list[str] = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "XOM",
        "JPM",
        "BA",
    ]
    data: MockData = MockData()
    pipeline: ResearchPipeline = ResearchPipeline(data)
    pipeline.portfolio_config.hard_factor_neutral = True
    pipeline.portfolio_config.neutral_factors = ("market_beta", "size")
    pipeline.portfolio_config.sector_neutral = True
    pipeline.portfolio_config.use_capacity_limits = True
    pipeline.portfolio_config.aum = 250_000_000.00
    # Optional: use smaller risk/turnover penalties so the optimizer takes larger
    # positions (otherwise notionals stay tiny vs AUM with small random alphas):
    pipeline.portfolio_config.risk_aversion = 0.05
    pipeline.portfolio_config.turnover_penalty = 0.01

    alpha_models: dict[str, dict[str, float]] = {
        "value": {t: float(np.random.normal(0, 0.02)) for t in tickers},
        "momentum": {t: float(np.random.normal(0, 0.02)) for t in tickers},
    }
    weights: dict[str, float]
    trades: dict[str, float]
    cost: float
    weights, trades, cost = pipeline.build_portfolio(alpha_models)
    logger.info("portfolio_built", weights=weights, cost=cost)

    aum: float = float(pipeline.portfolio_config.aum)
    latest_prices: pd.Series = data.get_prices(tickers).iloc[-1]
    shares = weights_to_shares(weights, aum, latest_prices)
    longs, shorts = shares_to_long_short(shares)
    # Dollar notionals (shares * price) for sanity check vs AUM
    long_notional = sum(longs[t] * latest_prices[t] for t in longs)
    short_notional = sum(shorts[t] * latest_prices[t] for t in shorts)
    logger.info(
        "position_shares",
        aum=aum,
        long_notional_usd=round(long_notional, 2),
        short_notional_usd=round(short_notional, 2),
        shares_long=longs,
        shares_short=shorts,
    )

    capacity_report: pd.DataFrame = pipeline.liquidity.capacity_report(
        weights, aum=pipeline.portfolio_config.aum
    )
    logger.info(
        "capacity_report", report_head=capacity_report.head().to_string()
    )

    ext: ResearchExtensions = ResearchExtensions(data)
    tracker: ExperimentRun = ext.experiments.start_run(
        "elite-demo", tags={"stage": "research"}
    )
    tracker.log_params(
        {"tickers": tickers, "portfolio_config": pipeline.portfolio_config}
    )
    tracker.log_metrics(
        {"cost": cost, "gross": sum(abs(v) for v in weights.values())}
    )
    tracker.log_artifact_json("weights.json", weights)
    tracker.finalize()

    returns_frame: pd.DataFrame = (
        data.get_prices(tickers).pct_change().dropna().tail(120)
    )
    vec: VectorizedBacktestResult = ext.vectorized.run(
        pd.DataFrame(
            np.random.normal(0, 1, (120, len(tickers))),
            index=pd.date_range("2022-01-01", periods=120, freq="B"),
            columns=tickers,
        ),
        returns_frame,
    )
    equity_tail = vec.equity_curve.tail()
    logger.info(
        "vectorized_backtest_equity_tail",
        equity_curve_tail={str(k): float(v) for k, v in equity_tail.items()},
    )

    # --- Optional demos: switch objective and run once ---
    # GMV (minimize variance only)
    pipeline.portfolio_config.objective_type = "gmv"
    w_gmv, _, _ = pipeline.build_portfolio(alpha_models)
    logger.info(
        "gmv_weights_sample", gmv_gross=sum(abs(v) for v in w_gmv.values())
    )

    # Tracking error vs benchmark
    pipeline.portfolio_config.objective_type = "tracking_error"
    pipeline.portfolio_config.benchmark_weights = {
        t: 1.0 / len(tickers) for t in tickers
    }
    w_te, _, _ = pipeline.build_portfolio(alpha_models)
    pipeline.portfolio_config.benchmark_weights = None

    # CVaR (downside risk)
    pipeline.portfolio_config.objective_type = "cvar"
    w_cvar, _, _ = pipeline.build_portfolio(alpha_models)
    pipeline.portfolio_config.objective_type = "mean_variance"

    # MPC first period (multi-period)
    if hasattr(pipeline, "build_portfolio_mpc"):
        w_mpc, trades_mpc, cost_mpc = pipeline.build_portfolio_mpc(
            alpha_models, mpc_horizon=3, mpc_discount=0.99
        )
        logger.info("mpc_first_period", cost=cost_mpc)

    # RL env (one reset + step)
    from ai_qre.backtest.portfolio_env import PortfolioEnv

    ret_df = data.get_returns(tickers).tail(60)

    def env_cost(trades: dict[str, float]) -> float:
        return 0.001 * sum(abs(x) for x in trades.values())

    env = PortfolioEnv(ret_df, tickers, env_cost)
    state, info = env.reset()
    action = {t: 0.0 for t in tickers}
    action[tickers[0]] = 0.02
    action[tickers[1]] = -0.02
    next_s, reward, term, trunc, step_info = env.step(action)
    logger.info("rl_env_step", reward=reward, pnl=step_info.get("pnl"))


if __name__ == "__main__":
    main()
