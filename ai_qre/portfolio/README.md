# portfolio

This folder contains the **portfolio optimizer**: the single-period convex optimization that turns alpha, risk (covariance), and constraints into target weights.

---

## Files

### `optimizer.py`

Single class: **`PortfolioOptimizer`**, which solves a quadratic program (QP) using CVXPY. Objective is to maximize alpha minus risk and turnover penalties, subject to linear constraints.

- **Constructor**: `PortfolioOptimizer(cov: ShrinkageCovariance, config: PortfolioConfig)`.

  - `cov`: used to compute the covariance matrix for the assets in each solve.
  - `config`: all constraint and penalty parameters (see `ai_qre.config.PortfolioConfig`).

- **`solve(alphas, current=None, factor_exposures=None, max_weight_by_asset=None) -> WeightVector`**

  - **Inputs**:
    - `alphas`: mapping ticker → float (combined alpha from the pipeline).
    - `current`: optional current weights (ticker → float); used for turnover. If None, turnover is vs zero.
    - `factor_exposures`: optional DataFrame (index = tickers, columns = factor names). If provided, a factor penalty and optional hard neutrality are applied.
    - `max_weight_by_asset`: optional Series or dict of ticker → max absolute weight. Tighter than the global `max_position` for specific names (e.g. from capacity).
  - **Objective** (maximized):
    - **Alpha**: `alpha_vec @ weights`.
    - **Risk**: `- risk_aversion * (weights' @ Sigma @ weights)`.
    - **Turnover**: `- turnover_penalty * sum(|weights - current|)`.
    - **Factor penalty** (if factor_exposures given): `- factor_penalty * sum((exposure' @ weights)²)` over factors.
  - **Constraints**:
    - **Net**: `sum(weights) == net_target` (often 0 for long/short).
    - **Gross**: `sum(|weights|) <= gross_limit`.
    - **Bounds**: `-upper_bounds <= weights <= upper_bounds`. Upper bounds are the minimum of `max_position` and, when provided, `max_weight_by_asset` per ticker.
    - **Hard factor/sector neutrality** (when configured): for selected columns in `factor_exposures` (from `neutral_factors` and/or columns starting with `sector_`), constraints `exposure' @ weights <= tol` and `>= -tol` with `factor_tolerance`.
  - **Post-processing**: If `max_names` is set in config, only the top `max_names` positions by absolute weight are kept; the rest are set to zero.
  - **Solver**: Uses `config.solver` (e.g. `"OSQP"`); falls back to OSQP if the name is unknown.
  - **Output**: Returns a `WeightVector` (dict ticker → float). If the solver fails, returns all zeros.

- **`_hard_neutral_columns(exposures)`**

  - Returns the list of exposure columns that get hard constraints: those in `config.neutral_factors` when `hard_factor_neutral` is True, plus all columns whose name starts with `sector_` when `sector_neutral` is True.

- **`_apply_max_names(weights)`**
  - If `config.max_names` is set and less than the number of names, keeps only the top `max_names` by absolute weight and zeros out the rest.

**Used by**: `ResearchPipeline.build_portfolio`, which constructs the optimizer with the pipeline’s covariance and config, passes blended/decayed/shrunk alpha, current weights, factor exposures (from Barra-like model when requested), and capacity-based max weights when enabled.
