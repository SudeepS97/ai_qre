from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class MeanVarianceInputs:
    alphas: Mapping[str, float]
    cov_matrix: np.ndarray
    current: Optional[Mapping[str, float]]
    risk_aversion: float
    turnover_penalty: float


@dataclass(frozen=True)
class GlobalMinimumVarianceInputs:
    cov_matrix: np.ndarray


@dataclass(frozen=True)
class TrackingErrorInputs:
    alphas: Mapping[str, float]
    cov_matrix: np.ndarray
    current: Optional[Mapping[str, float]]
    benchmark: Mapping[str, float]
    risk_aversion: float
    turnover_penalty: float


class BaseObjective:
    def build(
        self,
        tickers: Sequence[str],
        weights_var: cp.Variable,
    ) -> cp.Expression:
        raise NotImplementedError


class MeanVarianceObjective(BaseObjective):
    def __init__(self, inputs: MeanVarianceInputs) -> None:
        self.inputs = inputs

    def build(
        self,
        tickers: Sequence[str],
        weights_var: cp.Variable,
    ) -> cp.Expression:
        current_mapping: Mapping[str, float] = self.inputs.current or {}
        current_array = np.asarray(
            [float(current_mapping.get(ticker, 0.0)) for ticker in tickers],
            dtype=float,
        )
        alpha_vec = np.asarray(
            [float(self.inputs.alphas[ticker]) for ticker in tickers],
            dtype=float,
        )
        risk_term = cp.quad_form(weights_var, self.inputs.cov_matrix)
        turnover_term = cp.norm1(weights_var - current_array)
        return (
            alpha_vec @ weights_var
            - float(self.inputs.risk_aversion) * risk_term
            - float(self.inputs.turnover_penalty) * turnover_term
        )


class GlobalMinimumVarianceObjective(BaseObjective):
    def __init__(self, inputs: GlobalMinimumVarianceInputs) -> None:
        self.inputs = inputs

    def build(
        self,
        tickers: Sequence[str],
        weights_var: cp.Variable,
    ) -> cp.Expression:
        return -cp.quad_form(weights_var, self.inputs.cov_matrix)


class TrackingErrorObjective(BaseObjective):
    def __init__(self, inputs: TrackingErrorInputs) -> None:
        self.inputs = inputs

    def build(
        self,
        tickers: Sequence[str],
        weights_var: cp.Variable,
    ) -> cp.Expression:
        current_mapping: Mapping[str, float] = self.inputs.current or {}
        current_array = np.asarray(
            [float(current_mapping.get(ticker, 0.0)) for ticker in tickers],
            dtype=float,
        )
        alpha_vec = np.asarray(
            [float(self.inputs.alphas[ticker]) for ticker in tickers],
            dtype=float,
        )
        benchmark_array = np.asarray(
            [
                float(self.inputs.benchmark.get(ticker, 0.0))
                for ticker in tickers
            ],
            dtype=float,
        )
        active_weights = weights_var - benchmark_array
        risk_term = cp.quad_form(active_weights, self.inputs.cov_matrix)
        turnover_term = cp.norm1(weights_var - current_array)
        return (
            alpha_vec @ active_weights
            - float(self.inputs.risk_aversion) * risk_term
            - float(self.inputs.turnover_penalty) * turnover_term
        )
