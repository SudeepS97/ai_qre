# ai_qre package

This is the root of the **ai_qre** package. It holds the main entry points, configuration, types, and the core research pipeline. Subpackages (alpha, backtest, capacity, data, execution, portfolio, risk, tracking, stress, distributed, utils) each have their own **README.md** with a detailed explanation of the code in that folder.

---

## Root-level files

### `__init__.py`

Public API: exports **`ResearchPipeline`** and **`ResearchExtensions`**. These are the two main entry points for using the library.

---

### `config.py`

Dataclasses that hold **all configurable parameters** for the pipeline and extensions. No logic—only defaults. You mutate these (e.g. `pipeline.portfolio_config.max_position = 0.05`) to change behavior.

- **`PortfolioConfig`**: Optimizer and portfolio constraints.

  - Exposure: `max_position`, `gross_limit`, `net_target`, `turnover_penalty`, `risk_aversion`, `factor_penalty`, `borrow_cost_penalty`.
  - Neutrality: `sector_neutral`, `hard_factor_neutral`, `neutral_factors`, `factor_tolerance`.
  - Other: `max_names`, `solver`, `use_capacity_limits`, `aum`.
  - Objectives: `objective_type` (mean_variance, gmv, cvar, tracking_error, robust_mv), `benchmark_weights`, `uncertainty_radius`, `uncertainty_type`.
  - Resampled: `use_resampled_efficiency`, `resampled_simulations`, `resampled_seed`.
  - Black–Litterman: `use_black_litterman`, `bl_tau`, `bl_omega_scale`, `bl_views`.
  - Trading cost in objective: `use_trading_cost_in_objective`, `trading_cost_impact`; `turnover_limit`.

- **`RiskConfig`**: Covariance and factor model.

  - `shrinkage` (covariance), `factor_window`, `momentum_lookback`, `min_obs`.

- **`ExecutionConfig`**: Defaults for execution cost (spread, impact). Note: `ExecutionSimulator` uses its constructor args, not this dataclass.

- **`WalkForwardConfig`**: Walk-forward backtest: `train_window`, `test_window`, `step_size`, `rebalance_every`, `min_history`; `use_mpc`, `mpc_horizon`, `mpc_discount`.

- **`StressTestConfig`**: Monte Carlo stress: `paths`, `horizon`, `seed`.

- **`DistributedConfig`**: Parallel runner: `workers`, `chunksize`.

- **`CapacityConfig`**: Liquidity model: `adv_fraction`, `participation_cap`, `forecast_days_to_liquidate`, `min_weight_cap`.

- **`ExperimentConfig`**: Experiment tracker: `root_dir`, `autosave_metrics`, `autosave_params`, `autosave_artifacts`.

- **`VectorizedResearchConfig`**: Vectorized backtest: `rebalance_frequency`, `top_n`, `bottom_n`, `long_short`, `gross`, `neutralize_each_date`.

---

### `types.py`

**Type aliases** and **protocols** used across the package.

- **Aliases**: `Ticker`, `AlphaVector` (dict ticker → score), `AlphaModelMap` (dict model name → AlphaVector), `WeightVector`, `TradeVector`, `FactorExposureMap` (date → exposures DataFrame).
- **`CovarianceProvider`**: Protocol with `compute(tickers) -> np.ndarray`. Used by the optimizer and resampling.
- **`MarketDataProviderLike`**: Protocol with `get_prices`, `get_returns`, `get_volumes`, `get_sectors`, `get_market_caps`. Any implementation can be passed as the pipeline’s `data`.
- **`ResearchPipelineLike`**: Protocol with `build_portfolio(alpha_models, current?, alpha_age?, use_factor_penalty?)` returning `(WeightVector, TradeVector, float)`. Used by the walk-forward backtester.
- **`AlphaGeneratorLike`**: Protocol: callable that takes `train_returns: pd.DataFrame` and returns `AlphaModelMap`. Used by the walk-forward backtester to produce alphas from training data.

---

### `research_pipeline.py`

**`ResearchPipeline`** is the main production path: data + alpha models in, weights + trades + cost out.

- **Constructor**: `ResearchPipeline(data: MarketDataProvider)`.

  - Creates: `AlphaBlender`, `AlphaDecay`, `PortfolioConfig`, `RiskConfig`, `CapacityConfig`, `ShrinkageCovariance(data, risk_config.shrinkage)`, `BarraLikeRiskModel(data, risk_config)`, `LiquidityModel(data, capacity_config)`, `ExecutionSimulator()`.

- **`build_portfolio(alpha_models, current=None, alpha_age=0, use_factor_penalty=True) -> (WeightVector, TradeVector, float)`**:

  1. **Blend** alpha_models with the blender (default equal weights).
  2. **Decay** by alpha_age.
  3. **Shrink** toward 0.
  4. **(Optional) Black–Litterman**: if use_black_litterman and bl_views, replace alpha with posterior_expected_returns(tickers, alpha, cov, views, tau, omega_scale).
  5. Build **PortfolioOptimizer** with the pipeline’s cov and portfolio_config.
  6. **Factor exposures**: if use_factor_penalty and tickers non-empty, get Barra-like exposures for tickers.
  7. **Capacity**: if use_capacity_limits, get max_weight_by_asset from LiquidityModel using config.aum.
  8. **(Optional) Trading cost in objective**: if use_trading_cost_in_objective and current, get trading_cost_lambda_diag from LiquidityModel and pass to optimizer.
  9. **Solve** optimizer with alpha, current weights, factor_exposures, max_weight_by_asset, trading_cost_lambda_diag (optimizer may use resampled efficiency path when use_resampled_efficiency is True).
  10. Compute **trades** = new weights − current.
  11. **Cost** = ExecutionSimulator.portfolio_cost(trades).
  12. Return (weights, trades, cost).

- **`build_portfolio_mpc(alpha_models, current=None, alpha_age=0, use_factor_penalty=True, mpc_horizon=3, mpc_discount=0.99) -> (WeightVector, TradeVector, float)`**: Same blend/decay/shrink and optional Black–Litterman; then uses **solve_mpc_first_period** (multi-period) instead of the single-period optimizer. Returns the same (weights, trades, cost) shape for the first period.

You can mutate `pipeline.portfolio_config`, `pipeline.risk_config`, `pipeline.capacity_config` (and the blender/decay) to tune behavior.

---

### `research_extensions.py`

**`ResearchExtensions`** is a convenience facade that instantiates **advanced research utilities** that are not in the main pipeline path. Useful for backtesting, stress tests, experiments, and parallel sweeps.

- **Constructor**: `ResearchExtensions(data_provider: MarketDataProvider)`.

  - Creates: `BarraLikeRiskModel(data_provider)`, `CrossSectionalAlphaModel()`, `WalkForwardBacktester()`, `VectorizedResearchHarness()`, `MonteCarloStress()`, `DistributedResearchRunner()`, `ExperimentTracker()`.

- **Attributes**:
  - **`barra_risk`**: Factor exposures and risk snapshots.
  - **`alpha_regression`**: Cross-sectional OLS/ridge for signal → return.
  - **`walk_forward`**: Walk-forward backtester (needs pipeline + alpha_generator + data + tickers).
  - **`vectorized`**: Vectorized alpha backtest (alpha DataFrame + returns DataFrame).
  - **`stress`**: Monte Carlo stress (weights + returns → stats).
  - **`distributed`**: Multiprocessing run/starmap.
  - **`experiments`**: Experiment tracker (start_run, log params/metrics/artifacts).

Each of these is documented in the README of the corresponding subfolder (risk, alpha, backtest, stress, distributed, tracking).

---

## Subpackages (see each folder’s README)

| Folder           | README                                         | Contents                                                                                                               |
| ---------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **alpha/**       | [alpha/README.md](alpha/README.md)             | Alpha blending, decay, shrink, orthogonalize; cross-sectional regression.                                              |
| **backtest/**    | [backtest/README.md](backtest/README.md)       | Backtester, portfolio_env (PortfolioEnv, build_state, default_reward_fn), vectorized harness, walk-forward backtester. |
| **capacity/**    | [capacity/README.md](capacity/README.md)       | Liquidity model, ADDV, max weights, capacity report, trading_cost_impact_diag.                                         |
| **data/**        | [data/README.md](data/README.md)               | MarketDataProvider protocol.                                                                                           |
| **execution/**   | [execution/README.md](execution/README.md)     | Execution cost (spread + impact).                                                                                      |
| **portfolio/**   | [portfolio/README.md](portfolio/README.md)     | Portfolio optimizer, resampling, objectives, constraints, Black–Litterman, multi-period (see portfolio README).        |
| **risk/**        | [risk/README.md](risk/README.md)               | Shrinkage covariance, Barra-like model, simple factor model.                                                           |
| **tracking/**    | [tracking/README.md](tracking/README.md)       | Experiment run and tracker.                                                                                            |
| **stress/**      | [stress/README.md](stress/README.md)           | Monte Carlo stress.                                                                                                    |
| **distributed/** | [distributed/README.md](distributed/README.md) | Multiprocessing runner.                                                                                                |
| **utils/**       | [utils/README.md](utils/README.md)             | Position sizing (weights → shares), logging.                                                                           |

For a **granular understanding** of what each file does, open the README in the relevant subfolder.
