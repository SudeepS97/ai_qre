from collections.abc import Mapping

import cvxpy as cp
import numpy as np

from ai_qre.portfolio.objectives import (
    CvarInputs,
    CvarObjective,
    GlobalMinimumVarianceInputs,
    GlobalMinimumVarianceObjective,
    MeanVarianceInputs,
    MeanVarianceObjective,
    RobustMeanVarianceInputs,
    RobustMeanVarianceObjective,
)


def _zeros_like_mapping(keys: list[str]) -> Mapping[str, float]:
    return {key: 0.0 for key in keys}


def test_mean_variance_reduces_to_gmv_when_alphas_zero() -> None:
    tickers = ["A", "B"]
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    alphas = _zeros_like_mapping(tickers)

    mv_inputs = MeanVarianceInputs(
        alphas=alphas,
        cov_matrix=cov,
        current=None,
        risk_aversion=1.0,
        turnover_penalty=0.0,
    )
    gmv_inputs = GlobalMinimumVarianceInputs(cov_matrix=cov)

    weights = cp.Variable(len(tickers))

    mv_expr = MeanVarianceObjective(mv_inputs).build(tickers, weights)
    gmv_expr = GlobalMinimumVarianceObjective(gmv_inputs).build(
        tickers, weights
    )

    assert mv_expr is not None
    assert gmv_expr is not None


def test_cvar_objective_builds_expression() -> None:
    tickers = ["A", "B"]
    alphas = _zeros_like_mapping(tickers)
    scenario_returns = np.array(
        [[0.01, -0.02], [-0.03, 0.04], [0.0, -0.01]], dtype=float
    )
    inputs = CvarInputs(
        alphas=alphas,
        scenario_returns=scenario_returns,
        current=None,
        risk_aversion=1.0,
        turnover_penalty=0.0,
    )

    weights = cp.Variable(len(tickers))
    expr = CvarObjective(inputs).build(tickers, weights)
    assert expr is not None


def test_robust_mv_objective_builds_expression() -> None:
    tickers = ["A", "B"]
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    alphas = _zeros_like_mapping(tickers)
    base = MeanVarianceInputs(
        alphas=alphas,
        cov_matrix=cov,
        current=None,
        risk_aversion=1.0,
        turnover_penalty=0.0,
    )
    robust_inputs = RobustMeanVarianceInputs(
        base=base,
        uncertainty_radius=0.01,
        uncertainty_type="box",
    )

    weights = cp.Variable(len(tickers))
    expr = RobustMeanVarianceObjective(robust_inputs).build(tickers, weights)
    assert expr is not None
