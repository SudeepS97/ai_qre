
from dataclasses import dataclass

@dataclass
class PortfolioConfig:
    max_position: float = 0.03
    gross_limit: float = 3.0
    net_target: float = 0.0
    turnover_penalty: float = 0.1

@dataclass
class RiskConfig:
    shrinkage: float = 0.1
    factor_window: int = 252

@dataclass
class ExecutionConfig:
    spread_cost: float = 0.0005
    impact_coeff: float = 0.1
