"""Configuration dataclasses for portfolio, risk, backtest, and research."""

from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    """Portfolio optimizer and constraints: exposure, neutrality, objective type, BL, capacity."""

    max_position: float = 0.03
    gross_limit: float = 3.0
    net_target: float = 0.0
    turnover_penalty: float = 0.1
    risk_aversion: float = 0.5
    factor_penalty: float = 0.05
    borrow_cost_penalty: float = 0.0
    sector_neutral: bool = False
    max_names: int | None = None
    solver: str = "OSQP"
    objective_type: str = "mean_variance"
    benchmark_weights: dict[str, float] | None = None
    uncertainty_radius: float = 0.0
    uncertainty_type: str = "box"
    use_resampled_efficiency: bool = False
    resampled_simulations: int = 50
    resampled_seed: int | None = None
    use_black_litterman: bool = False
    bl_tau: float = 0.025
    bl_omega_scale: float = 1.0
    bl_views: tuple[tuple[str, tuple[tuple[str, float], ...], float], ...] = ()
    hard_factor_neutral: bool = False
    neutral_factors: tuple[str, ...] = ("market_beta",)
    factor_tolerance: float = 1e-8
    use_capacity_limits: bool = False
    aum: float = 100_000_000.0
    use_trading_cost_in_objective: bool = False
    trading_cost_impact: float = 0.1
    turnover_limit: float | None = None


@dataclass
class RiskConfig:
    """Covariance and factor model: shrinkage, factor window, momentum lookback."""

    shrinkage: float = 0.1
    factor_window: int = 252
    momentum_lookback: int = 126
    min_obs: int = 60


@dataclass
class ExecutionConfig:
    """Execution cost model: spread and impact coefficients."""

    spread_cost: float = 0.0005
    impact_coeff: float = 0.1


@dataclass
class WalkForwardConfig:
    """Walk-forward backtest: train/test windows, step size, optional MPC."""

    train_window: int = 252
    test_window: int = 21
    step_size: int = 21
    rebalance_every: int = 21
    min_history: int = 252
    use_mpc: bool = False
    mpc_horizon: int = 3
    mpc_discount: float = 0.99


@dataclass
class StressTestConfig:
    """Monte Carlo stress: number of paths, horizon, seed."""

    paths: int = 1000
    horizon: int = 50
    seed: int | None = 7


@dataclass
class DistributedConfig:
    """Multiprocessing runner: workers, chunksize."""

    workers: int | None = None
    chunksize: int = 1


@dataclass
class CapacityConfig:
    """Liquidity model: participation cap, days to liquidate, min weight cap."""

    adv_fraction: float = 0.05
    participation_cap: float = 0.05
    forecast_days_to_liquidate: int = 5
    min_weight_cap: float = 0.0


@dataclass
class ExperimentConfig:
    """Experiment tracker: root dir, autosave flags."""

    root_dir: str = "runs"
    autosave_metrics: bool = True
    autosave_params: bool = True
    autosave_artifacts: bool = True


@dataclass
class VectorizedResearchConfig:
    """Vectorized backtest: rebalance frequency, top_n/bottom_n, gross, neutralization."""

    rebalance_frequency: int = 21
    top_n: int | None = None
    bottom_n: int | None = None
    long_short: bool = True
    gross: float = 1.0
    neutralize_each_date: bool = False
