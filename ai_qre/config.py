from dataclasses import dataclass


@dataclass
class PortfolioConfig:
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
    hard_factor_neutral: bool = False
    neutral_factors: tuple[str, ...] = ("market_beta",)
    factor_tolerance: float = 1e-8
    use_capacity_limits: bool = False
    aum: float = 100_000_000.0


@dataclass
class RiskConfig:
    shrinkage: float = 0.1
    factor_window: int = 252
    momentum_lookback: int = 126
    min_obs: int = 60


@dataclass
class ExecutionConfig:
    spread_cost: float = 0.0005
    impact_coeff: float = 0.1


@dataclass
class WalkForwardConfig:
    train_window: int = 252
    test_window: int = 21
    step_size: int = 21
    rebalance_every: int = 21
    min_history: int = 252


@dataclass
class StressTestConfig:
    paths: int = 1000
    horizon: int = 50
    seed: int | None = 7


@dataclass
class DistributedConfig:
    workers: int | None = None
    chunksize: int = 1


@dataclass
class CapacityConfig:
    adv_fraction: float = 0.05
    participation_cap: float = 0.05
    forecast_days_to_liquidate: int = 5
    min_weight_cap: float = 0.0


@dataclass
class ExperimentConfig:
    root_dir: str = "runs"
    autosave_metrics: bool = True
    autosave_params: bool = True
    autosave_artifacts: bool = True


@dataclass
class VectorizedResearchConfig:
    rebalance_frequency: int = 21
    top_n: int | None = None
    bottom_n: int | None = None
    long_short: bool = True
    gross: float = 1.0
    neutralize_each_date: bool = False
