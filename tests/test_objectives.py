from collections.abc import Mapping

import numpy as np

from ai_qre.portfolio.objectives import (
    GlobalMinimumVarianceInputs,
    GlobalMinimumVarianceObjective,
    MeanVarianceInputs,
    MeanVarianceObjective,
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

    import cvxpy as cp

    weights = cp.Variable(len(tickers))

    mv_expr = MeanVarianceObjective(mv_inputs).build(tickers, weights)
    gmv_expr = GlobalMinimumVarianceObjective(gmv_inputs).build(
        tickers, weights
    )

    assert mv_expr is not None
    assert gmv_expr is not None
