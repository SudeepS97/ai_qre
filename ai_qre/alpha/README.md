# alpha

This folder contains code for **alpha signal construction, combination, and transformation**. Alpha is the expected excess return (or score) per asset that the portfolio optimizer uses as the objective to maximize.

---

## Files

### `transforms.py`

Utilities for combining and transforming alpha vectors before they are passed to the optimizer.

- **`AlphaBlender`**
  Combines multiple alpha models into a single alpha vector.

  - **Constructor**: `AlphaBlender(weights=None)`. `weights` is an optional mapping of model name → weight (e.g. `{"value": 0.6, "momentum": 0.4}`). If a model name is missing, its weight defaults to `1.0`.
  - **`blend(alpha_models: AlphaModelMap) -> AlphaVector`**: For each model, multiplies each ticker’s score by that model’s weight and sums across models per ticker. Returns one `dict[ticker, float]` (an `AlphaVector`).
  - Used by `ResearchPipeline` to merge the `alpha_models` dict before decay and shrink.

- **`AlphaDecay`**
  Applies time decay to alpha so that older signals have less influence.

  - **Constructor**: `AlphaDecay(half_life=5.0)`. `half_life` is in days; must be positive.
  - **`apply(alpha: AlphaVector, age_days: int | float) -> AlphaVector`**: Multiplies every score by `0.5 ** (age_days / half_life)`. So at `age_days == half_life`, alpha is scaled by 0.5.
  - The pipeline calls this when you pass `alpha_age` into `build_portfolio`.

- **`shrink(alpha, prior_mean=0.0, strength=0.5) -> AlphaVector`**
  Shrinks each alpha value toward a prior (default 0).

  - Formula: `(1 - strength) * value + strength * prior_mean`. `strength` is clamped to [0, 1].
  - Reduces extreme scores and can improve stability. Called by the pipeline after blend and decay.

- **`orthogonalize(alpha_matrix: np.ndarray) -> np.ndarray`**
  Pure NumPy helper: takes a 2D array (rows = observations, columns = alpha/signal series) and returns an orthogonalized matrix via QR decomposition (`Q` from `np.linalg.qr(alpha_matrix.T)`). Useful for building uncorrelated signal streams; not used inside the main pipeline.

**Types**: Uses `AlphaModelMap` and `AlphaVector` from `ai_qre.types`.

---

### `cross_sectional_regression.py`

Cross-sectional regression for predicting future returns from signals (e.g. to produce an alpha vector from a feature matrix).

- **`RegressionResult`** (dataclass)
  Holds outputs of a single fit: `coefficients` (pd.Series), `intercept` (float), `r2`, `fitted` (pd.Series), `residuals` (pd.Series).

- **`CrossSectionalAlphaModel`**
  Fits a cross-sectional OLS regression with optional ridge (L2) regularization.
  - **Constructor**: `CrossSectionalAlphaModel(ridge=1e-6)`. Ridge is added to the normal equations (except intercept) for stability.
  - **`fit(signal_df, future_returns, sample_weight=None) -> RegressionResult`**:
    - `signal_df`: DataFrame, index = entities (e.g. tickers or dates), columns = signal names.
    - `future_returns`: Series of forward returns, aligned to `signal_df` index.
    - Drops rows with any NaN in signals or target. Builds design matrix with intercept, optionally applies sqrt(sample*weight) to rows, solves `(X'X + ridge*I)_beta = X'y`(intercept column not penalized). Stores`coefficients_`, `intercept\_`, `columns`, and `last_result`.
  - **`predict(signal_df) -> pd.Series`**: Uses stored coefficients and intercept to compute predicted return for each row. Must call `fit` first.
  - You can use `predict()` on new data to get an alpha vector (ticker → predicted return) for the pipeline.

**Exposed in**: `ResearchExtensions.alpha_regression` and in `ai_qre.alpha.__init__`.

---

### `__init__.py`

Re-exports: `CrossSectionalAlphaModel`, `RegressionResult`, `AlphaBlender`, `AlphaDecay`, `orthogonalize`, `shrink`.
