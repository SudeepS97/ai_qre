# portfolio

This folder contains the **portfolio optimizer**: single-period convex optimization (and resampled efficiency) that turns alpha, risk (covariance), and constraints into target weights. It also provides Black–Litterman alpha adjustment, multi-period (MPC) first-period solve, objective builders, and constraints.

---

## Files

### `optimizer.py`

Single class: **`PortfolioOptimizer`**, which solves a quadratic program (QP) using CVXPY (or delegates to resampled efficiency when configured).

- **Constructor**: `PortfolioOptimizer(cov: CovarianceProvider, config: PortfolioConfig)`.

  - `cov`: any type implementing `CovarianceProvider` (e.g. `ShrinkageCovariance`); used to compute the covariance matrix for the assets in each solve.
  - `config`: all constraint and penalty parameters (see `ai_qre.config.PortfolioConfig`).

- **`solve(alphas, current=None, factor_exposures=None, max_weight_by_asset=None, trading_cost_lambda_diag=None) -> WeightVector`**

  - **Inputs**:
    - `alphas`: mapping ticker → float (combined alpha from the pipeline).
    - `current`: optional current weights (ticker → float); used for turnover. If None, turnover is vs zero.
    - `factor_exposures`: optional DataFrame (index = tickers, columns = factor names). If provided, a factor penalty and optional hard neutrality are applied.
    - `max_weight_by_asset`: optional Series or dict of ticker → max absolute weight. Tighter than the global `max_position` for specific names (e.g. from capacity).
    - `trading_cost_lambda_diag`: optional mapping ticker → float; per-asset quadratic trading cost coefficient in the objective (e.g. from `LiquidityModel.trading_cost_impact_diag`).
  - **Resampled path**: When `config.use_resampled_efficiency` is True, `solve()` delegates to `resampled_efficiency_weights(...)` and returns those weights (no CVXPY in that path).
  - **Objective type** (`config.objective_type`): `"mean_variance"` (default), `"gmv"`, `"cvar"`, `"tracking_error"`, `"robust_mv"`. Objectives are built from `objectives.py`; constraints from `basic_exposure_constraints` in `constraints.py` (net, gross, bounds, optional turnover limit).
  - **Objective** (mean-variance form): Alpha term, risk term, turnover penalty, factor penalty (if factor_exposures given); optionally trading cost quadratic term when `use_trading_cost_in_objective` and `trading_cost_lambda_diag` provided.
  - **Constraints**: Net, gross, per-asset bounds, hard factor/sector neutrality when configured; optional turnover limit when `config.turnover_limit` and `current` are set.
  - **Post-processing**: If `max_names` is set, only the top `max_names` positions by absolute weight are kept.
  - **Solver**: Uses `config.solver` (e.g. `"OSQP"`). Returns `WeightVector`; all zeros on solver failure.

- **`_hard_neutral_columns(exposures)`**, **`_apply_max_names(weights)`**: Internal helpers for hard neutrality columns and max-names post-processing.

**Used by**: `ResearchPipeline.build_portfolio`, which constructs the optimizer with the pipeline’s covariance and config, passes blended/decayed/shrunk (and optionally Black–Litterman) alpha, current weights, factor exposures, capacity-based max weights, and optionally `trading_cost_lambda_diag`.

---

### `resampling.py`

**Resampled efficiency (Michaud)** by averaging portfolio weights over bootstrap resamples of (μ, Σ).

- **`resampled_efficiency_weights(cov, config, tickers, alphas, optimizer_factory, current=None, n_simulations=None, seed=None) -> WeightVector`**
  - `optimizer_factory`: callable `(CovarianceProvider, PortfolioConfig) -> instance with solve(alphas, current=...)`. The optimizer uses `PortfolioOptimizer` as this factory when `use_resampled_efficiency` is True.
  - Runs multiple simulations with resampled returns/covariance and alpha; each solve returns a weight vector; result is the average over simulations.
  - Used by `PortfolioOptimizer.solve` when `config.use_resampled_efficiency` is True.

---

### `black_litterman.py`

**Black–Litterman** adjustment of expected returns given views.

- **`posterior_expected_returns(tickers, alpha, cov_matrix, views, tau, omega_scale) -> AlphaVector`**
  - Combines prior alpha with views to produce posterior expected returns (alpha) for the optimizer.
  - Used by the pipeline before the optimizer when `use_black_litterman` and `bl_views` are set.

---

### `multi_period.py`

**Multi-period (MPC)** optimization: solve for a horizon and return first-period weights.

- **`solve_mpc_first_period(tickers, alpha, cov_matrix, current, config, horizon, discount) -> WeightVector`**
  - Solves the multi-period problem and returns the first period’s optimal weights.
  - Used by `ResearchPipeline.build_portfolio_mpc`.

---

### `objectives.py`

**Objective builders** for the optimizer: input dataclasses and CVXPY objective constructors.

- **Input dataclasses**: `MeanVarianceInputs`, `GlobalMinimumVarianceInputs`, `TrackingErrorInputs`, `CvarInputs`, `RobustMeanVarianceInputs` (and robust base).
- **Objective classes**: `MeanVarianceObjective`, `GlobalMinimumVarianceObjective`, `TrackingErrorObjective`, `CvarObjective`, `RobustMeanVarianceObjective`; each has a `build(tickers, weights_var)` (or similar) that returns the CVXPY expression.
  - Used by `PortfolioOptimizer` to build the objective from `config.objective_type`.

---

### `constraints.py`

**CVXPY constraints** for net/gross exposure, bounds, and optional turnover limit.

- **`basic_exposure_constraints(config, tickers, weights_var, max_weight_by_asset, current=None) -> list[Constraint]`**
  - Net target, gross limit, per-asset upper/lower bounds (from `max_position` and optional per-asset caps).
  - If `config.turnover_limit` is set and `current` is not None, adds a turnover (L1 of weight change) constraint.
  - Used by `PortfolioOptimizer.solve`.
