from collections.abc import Mapping

import numpy as np
import pandas as pd


class MonteCarloStress:
    def __init__(self, seed: int | None = 7) -> None:
        self.seed = seed

    def simulate(
        self,
        weights: Mapping[str, float],
        returns: pd.DataFrame,
        paths: int = 1000,
        horizon: int = 50,
    ) -> dict[str, float]:
        tickers = list(returns.columns)
        weight_vector = np.asarray(
            [float(weights.get(ticker, 0.0)) for ticker in tickers],
            dtype=float,
        )
        mean_vector = returns.mean().to_numpy(dtype=float)
        covariance = returns.cov().to_numpy(dtype=float)
        rng = np.random.default_rng(self.seed)

        terminal_values = np.empty(paths, dtype=float)
        worst_drawdowns = np.empty(paths, dtype=float)
        value_at_risk_95 = np.empty(paths, dtype=float)
        conditional_var_95 = np.empty(paths, dtype=float)

        for index in range(paths):
            simulated_returns = rng.multivariate_normal(
                mean_vector, covariance, size=horizon
            )
            pnl = simulated_returns @ weight_vector
            equity_curve: np.ndarray = np.cumprod(1.0 + pnl)
            rolling_peak: np.ndarray = np.maximum.accumulate(equity_curve)
            drawdowns: np.ndarray = equity_curve / rolling_peak - 1.0
            terminal_values[index] = float(equity_curve[-1])
            worst_drawdowns[index] = float(drawdowns.min())
            losses: np.ndarray = -(equity_curve - 1.0)
            threshold = float(np.quantile(losses, 0.95))
            value_at_risk_95[index] = threshold
            tail_losses: np.ndarray = losses[losses >= threshold]
            conditional_var_95[index] = float(
                tail_losses.mean() if tail_losses.size else threshold
            )

        return {
            "mean_terminal": float(terminal_values.mean()),
            "median_terminal": float(np.median(terminal_values)),
            "p05_terminal": float(np.quantile(terminal_values, 0.05)),
            "p95_terminal": float(np.quantile(terminal_values, 0.95)),
            "worst_terminal": float(terminal_values.min()),
            "best_terminal": float(terminal_values.max()),
            "mean_max_drawdown": float(worst_drawdowns.mean()),
            "worst_max_drawdown": float(worst_drawdowns.min()),
            "mean_var_95": float(value_at_risk_95.mean()),
            "mean_cvar_95": float(conditional_var_95.mean()),
        }
