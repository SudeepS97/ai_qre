# risk

This folder contains **risk and covariance** components: how asset covariance is estimated and how factor exposures (e.g. Barra-style) are computed for use in the optimizer and in backtests.

---

## Files

### `covariance.py`

- **`ShrinkageCovariance`**
  Builds the **covariance matrix** used by the portfolio optimizer (quadratic risk term). Implements the **CovarianceProvider** protocol (`compute(tickers) -> np.ndarray`).
  - **Constructor**: `ShrinkageCovariance(data: MarketDataProvider, shrinkage=0.1)`. `shrinkage` must be in [0, 1].
  - **`compute(tickers) -> np.ndarray`**:
    - Fetches historical returns via `data.get_returns(tickers)`.
    - Sample covariance = `returns.cov()` (NumPy).
    - Shrinks toward a diagonal matrix: `(1 - shrinkage) * sample_cov + shrinkage * diag(sample_cov)`.
    - Returns a square matrix of shape `(len(tickers), len(tickers))` in the same order as `tickers`.
  - Shrinkage stabilizes the estimate when the number of observations is limited. Used by `PortfolioOptimizer` and by `ResearchPipeline` (which creates a single `ShrinkageCovariance` at init).

---

### `barra_model.py`

Barra-style **factor risk model**: exposures (market, size, momentum, sectors) and optional factor covariance / full asset covariance.

- **`FactorRiskSnapshot`** (frozen dataclass)

  - `exposures`: DataFrame (assets × factors).
  - `factor_cov`: DataFrame (factor covariance).
  - `idiosyncratic_var`: Series (per-asset residual variance).
  - `asset_cov`: DataFrame (full asset covariance = factor component + diag(idio)).

- **`BarraLikeRiskModel`**
  - **Constructor**: `BarraLikeRiskModel(data: MarketDataProvider, config: RiskConfig | None = None)`. Uses `RiskConfig` for `factor_window`, `momentum_lookback`, etc.
  - **Factors** (fixed set):
    - **market_beta**: regression beta of each asset’s return on the (equal-weight) market return over `factor_window`.
    - **size**: z-score of log(market_cap); market caps from `data.get_market_caps`.
    - **momentum**: z-score of trailing return over `momentum_lookback` (from prices).
    - **sector\_\***: one dummy per sector from `data.get_sectors` (e.g. `sector_Technology`).
  - **`compute_factor_exposures(tickers) -> pd.DataFrame`**: Returns a DataFrame with index = tickers, columns = factor names (market*beta, size, momentum, sector*\*). Used by the pipeline to pass into the optimizer (factor penalty and hard neutrality) and by the vectorized harness when neutralizing by date.
  - **`factor_covariance(tickers) -> pd.DataFrame`**: Estimates factor returns each period via pseudo-inverse of exposures × asset returns; then returns the covariance of those factor returns.
  - **`snapshot(tickers) -> FactorRiskSnapshot`**: Computes exposures, factor covariance, then idiosyncratic variance as max(0, diag(sample_cov - X @ factor_cov @ X')). Builds full asset covariance = X @ factor_cov @ X' + diag(idio).
  - **`_zscore(values)`**: Static helper: (values - mean) / std; returns 0 if std <= 0.

**Used by**: `ResearchPipeline` (factor exposures for optimizer and optional neutralization); `ResearchExtensions.barra_risk` for direct use.

---

### `factor_model.py`

- **`SimpleFactorModel`**
  Minimal **single-factor (market)** model.
  - **Constructor**: `SimpleFactorModel(data: MarketDataProvider)`.
  - **`estimate(tickers) -> dict[str, float]`**: For each ticker, beta = cov(asset return, market return) / var(market return), where “market” is the equal-weight average of returns. Returns dict ticker → beta.
  - Not used by the main pipeline; available for simple beta exposure or research.

---

### `__init__.py`

Exports: `BarraLikeRiskModel`, `FactorRiskSnapshot`, `ShrinkageCovariance`, `SimpleFactorModel`.
