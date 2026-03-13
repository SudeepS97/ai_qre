from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegressionResult:
    coefficients: pd.Series
    intercept: float
    r2: float
    fitted: pd.Series
    residuals: pd.Series


class CrossSectionalAlphaModel:
    """Simple cross-sectional OLS with optional ridge stabilization."""

    def __init__(self, ridge: float = 1e-6) -> None:
        self.ridge = ridge
        self.columns: list[str] = []
        self.coefficients_: pd.Series | None = None
        self.intercept_: float = 0.0
        self.last_result: RegressionResult | None = None

    def fit(
        self,
        signal_df: pd.DataFrame,
        future_returns: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> RegressionResult:
        x = signal_df.astype(float).copy()
        y = future_returns.reindex(x.index).astype(float)
        mask = x.notna().all(axis=1) & y.notna()
        x = x.loc[mask]
        y = y.loc[mask]
        if x.empty:
            raise ValueError(
                "No valid rows available for cross-sectional regression fit."
            )

        self.columns = list(x.columns)
        x_mat = x.to_numpy()
        y_vec = y.to_numpy()
        ones = np.ones((len(x_mat), 1))
        design = np.hstack([ones, x_mat])

        if sample_weight is not None:
            w = sample_weight.reindex(x.index).fillna(0.0).to_numpy()
            sqrt_w = np.sqrt(w)[:, None]
            design = design * sqrt_w
            y_vec = y_vec * sqrt_w.ravel()

        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_vec
        intercept = float(beta[0])
        coefs = pd.Series(beta[1:], index=self.columns)

        fitted = pd.Series(
            intercept + x.to_numpy() @ coefs.to_numpy(), index=x.index
        )
        residuals = y - fitted
        denom = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((residuals**2).sum()) / denom if denom > 0 else 0.0

        self.coefficients_ = coefs
        self.intercept_ = intercept
        result = RegressionResult(
            coefficients=coefs,
            intercept=intercept,
            r2=r2,
            fitted=fitted,
            residuals=residuals,
        )
        self.last_result = result
        return result

    def predict(self, signal_df: pd.DataFrame) -> pd.Series:
        coefs = self.coefficients_
        if coefs is None:
            raise ValueError("Model must be fit before calling predict().")
        x = signal_df.reindex(columns=self.columns).fillna(0.0).astype(float)
        preds = self.intercept_ + x.to_numpy() @ coefs.to_numpy()
        return pd.Series(preds, index=x.index, name="predicted_return")
