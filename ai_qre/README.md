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

- **`RiskConfig`**: Covariance and factor model.

  - `shrinkage` (covariance), `factor_window`, `momentum_lookback`, `min_obs`.

- **`ExecutionConfig`**: Defaults for execution cost (spread, impact). Note: `ExecutionSimulator` uses its constructor args, not this dataclass.

- **`WalkForwardConfig`**: Walk-forward backtest: `train_window`, `test_window`, `step_size`, `rebalance_every`, `min_history`.

- **`StressTestConfig`**: Monte Carlo stress: `paths`, `horizon`, `seed`.

- **`DistributedConfig`**: Parallel runner: `workers`, `chunksize`.

- **`CapacityConfig`**: Liquidity model: `adv_fraction`, `participation_cap`, `forecast_days_to_liquidate`, `min_weight_cap`.

- **`ExperimentConfig`**: Experiment tracker: `root_dir`, `autosave_metrics`, `autosave_params`, `autosave_artifacts`.

- **`VectorizedResearchConfig`**: Vectorized backtest: `rebalance_frequency`, `top_n`, `bottom_n`, `long_short`, `gross`, `neutralize_each_date`.

---

### `types.py`

**Type aliases** and **protocols** used across the package.

- **Aliases**: `Ticker`, `AlphaVector` (dict ticker → score), `AlphaModelMap` (dict model name → AlphaVector), `WeightVector`, `TradeVector`, `FactorExposureMap` (date → exposures DataFrame).
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
  4. Build **PortfolioOptimizer** with the pipeline’s cov and portfolio_config.
  5. **Factor exposures**: if use_factor_penalty and tickers non-empty, get Barra-like exposures for tickers.
  6. **Capacity**: if use_capacity_limits, get max_weight_by_asset from LiquidityModel using config.aum.
  7. **Solve** optimizer with alpha, current weights, factor_exposures, max_weight_by_asset.
  8. Compute **trades** = new weights − current.
  9. **Cost** = ExecutionSimulator.portfolio_cost(trades).
  10. Return (weights, trades, cost).

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

| Folder           | README                                         | Contents                                                                  |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------- |
| **alpha/**       | [alpha/README.md](alpha/README.md)             | Alpha blending, decay, shrink, orthogonalize; cross-sectional regression. |
| **backtest/**    | [backtest/README.md](backtest/README.md)       | Backtester, vectorized harness, walk-forward backtester.                  |
| **capacity/**    | [capacity/README.md](capacity/README.md)       | Liquidity model, ADDV, max weights, capacity report.                      |
| **data/**        | [data/README.md](data/README.md)               | MarketDataProvider protocol.                                              |
| **execution/**   | [execution/README.md](execution/README.md)     | Execution cost (spread + impact).                                         |
| **portfolio/**   | [portfolio/README.md](portfolio/README.md)     | Portfolio optimizer (QP).                                                 |
| **risk/**        | [risk/README.md](risk/README.md)               | Shrinkage covariance, Barra-like model, simple factor model.              |
| **tracking/**    | [tracking/README.md](tracking/README.md)       | Experiment run and tracker.                                               |
| **stress/**      | [stress/README.md](stress/README.md)           | Monte Carlo stress.                                                       |
| **distributed/** | [distributed/README.md](distributed/README.md) | Multiprocessing runner.                                                   |
| **utils/**       | [utils/README.md](utils/README.md)             | Position sizing (weights → shares), logging.                              |

For a **granular understanding** of what each file does, open the README in the relevant subfolder.
