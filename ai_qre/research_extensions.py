from ai_qre.alpha.cross_sectional_regression import CrossSectionalAlphaModel
from ai_qre.backtest.vectorized import (
    VectorizedBacktestResult,
    VectorizedResearchHarness,
)
from ai_qre.backtest.walk_forward import WalkForwardBacktester
from ai_qre.data.provider import MarketDataProvider
from ai_qre.distributed.runner import DistributedResearchRunner
from ai_qre.risk.barra_model import BarraLikeRiskModel
from ai_qre.stress.monte_carlo import MonteCarloStress
from ai_qre.tracking.experiment import ExperimentRun, ExperimentTracker


class ResearchExtensions:
    """Facade over backtest (vectorized, walk_forward), stress, experiments, distributed runner, Barra risk, cross-sectional alpha."""

    def __init__(self, data_provider: MarketDataProvider) -> None:
        self.barra_risk = BarraLikeRiskModel(data_provider)
        self.alpha_regression = CrossSectionalAlphaModel()
        self.walk_forward = WalkForwardBacktester()
        self.vectorized = VectorizedResearchHarness()
        self.stress = MonteCarloStress()
        self.distributed = DistributedResearchRunner()
        self.experiments = ExperimentTracker()


__all__ = [
    "ResearchExtensions",
    "ExperimentRun",
    "ExperimentTracker",
    "VectorizedBacktestResult",
    "VectorizedResearchHarness",
]
