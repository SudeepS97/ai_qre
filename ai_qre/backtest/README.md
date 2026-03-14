# backtest

This folder contains **backtesting** components: from simple static-weight backtests to vectorized alpha-backtests and full walk-forward simulations that use the research pipeline.

---

## Files

### `backtester.py`

Minimal backtester for a **single, static** weight vector over a return series.

- **`Backtester`**
  - **`run(weights: Mapping[str, float], returns: pd.DataFrame) -> pd.Series`**:
    - `weights`: ticker → weight (e.g. from `build_portfolio`).
    - `returns`: DataFrame with dates as index and tickers as columns.
    - For each date, portfolio PnL = sum over tickers of `weight[ticker] * return[ticker]`.
    - Returns the **cumulative product of (1 + daily PnL)** as a Series (i.e. a gross equity curve starting at 1).
  - No rebalancing, no costs; useful for quick sanity checks of a fixed portfolio vs history.

---

### `vectorized.py`

Vectorized backtest that turns a **time series of alpha scores** into rebalanced weights and an equity curve. Does not use the full pipeline (no optimizer, no execution cost); it’s for fast alpha evaluation.

- **`VectorizedBacktestResult`** (frozen dataclass)

  - `weights`: DataFrame (index = rebalance dates, columns = tickers).
  - `portfolio_returns`: Series of daily portfolio returns.
  - `equity_curve`: Series, cumulative product of (1 + portfolio_returns).
  - `turnover`: Series of turnover at each rebalance (L1 change in weights).

- **`VectorizedResearchHarness`**
  - **Constructor**: `VectorizedResearchHarness(config=None)`. Uses `VectorizedResearchConfig` (rebalance_frequency, top_n, bottom_n, long_short, gross, neutralize_each_date).
  - **`run(alpha_frame, returns_frame, factor_exposure_by_date=None) -> VectorizedBacktestResult`**:
    - `alpha_frame`: DataFrame, index = dates, columns = tickers, values = alpha scores.
    - `returns_frame`: DataFrame, same index/columns, daily returns.
    - At each rebalance date (every `rebalance_frequency` days), scores are converted to weights by `_weights_from_scores`:
      - If `top_n` is set: long top N, else long all with score > 0.
      - If `long_short` and `bottom_n` set: short bottom N, else short all with score < 0.
      - Weights are +1 / -1 on those names, 0 elsewhere.
    - If `neutralize_each_date` and `factor_exposure_by_date` is provided, weights are orthogonalized to the given factor exposures (regression-style neutralization).
    - Weights are scaled to a target gross exposure via `_scale_to_gross(weights, gross)`.
    - Between rebalance dates, weights are forward-filled. Portfolio return each day = sum(weight \* return).
    - Turnover at each rebalance = sum of absolute weight changes.
  - **`_weights_from_scores`**: Implements the long/short and top_n/bottom_n logic.
  - **`_neutralize`**: Projects weights onto the null space of the exposure matrix (factor-neutral weights).
  - **`_scale_to_gross`**: Multiplies weights so that sum of absolute weights equals the configured gross.

**Exposed in**: `ResearchExtensions.vectorized` and `ai_qre.backtest.__init__`.

---

### `portfolio_env.py`

**RL-style portfolio environment** for use with external reinforcement learning agents: reset → state, step(action) → next state, reward, done, info.

- **`PortfolioEnv`**

  - **Constructor**: `PortfolioEnv(returns, tickers, cost_fn, state_window=21, reward_fn=default_reward_fn)`.
    - `returns`: DataFrame (dates × tickers).
    - `tickers`: list of ticker names.
    - `cost_fn`: callable `(TradeVector) -> float` (e.g. execution cost).
    - `state_window`: number of periods of returns in state.
    - `reward_fn`: callable `(pnl, cost, state, action, info) -> float`; default is PnL minus cost.
  - **`reset()`**: Returns initial state (positions, returns window, date index, tickers).
  - **`step(action_weights: WeightVector)`**: Applies action as new weights; returns next_state, reward, done, info. State includes positions and a window of returns.

- **`build_state(positions, returns_window, date_idx, tickers) -> dict`**: Helper to build the standard state dict for RL.

- **`default_reward_fn(pnl, cost, state, action, info) -> float`**: Reward = PnL − cost.

**Exposed in**: `ai_qre.backtest.__init__` (`PortfolioEnv`, `build_state`, `default_reward_fn`).

---

### `walk_forward.py`

**Walk-forward backtest** that uses the full research pipeline and an alpha generator over rolling train/test windows. Simulates out-of-sample rebalancing with costs.

- **`WalkForwardResult`** (frozen dataclass)

  - `equity_curve`: Series, stitched NAV over time.
  - `weights_by_rebalance`: DataFrame, rebalance date × ticker.
  - `turnover_by_rebalance`: Series.
  - `cost_by_rebalance`: Series (execution cost at each rebalance).

- **`WalkForwardBacktester`**
  - **Constructor**: `WalkForwardBacktester(config=None)`. Uses `WalkForwardConfig`: train_window, test_window, step_size, rebalance_every, min_history; optionally use_mpc, mpc_horizon, mpc_discount for callers that use an MPC pipeline.
  - **`run(pipeline, alpha_generator, data_provider, tickers) -> WalkForwardResult`**:
    - Gets prices from `data_provider.get_prices(tickers)`, then returns = pct_change.
    - For each step: train slice = `returns[start - train_window : start]`, test slice = `returns[start : start + test_window]`.
    - Skips if train length < min_history or test is empty.
    - Calls `alpha_generator(train_slice)` to get `AlphaModelMap`, then `pipeline.build_portfolio(alpha_models, current=current, alpha_age=0)` to get weights, trades, cost.
    - Records weights at the first date of the test window, turnover (sum of |trades|), and cost.
    - Portfolio PnL on test = sum(weight \* return) per day; cost is subtracted from the first day’s PnL.
    - Builds a segment equity curve (cumprod of 1 + daily PnL), scales by running NAV, and concatenates segments (deduplicating index with keep="last").
    - `current` is updated to the new weights for the next step (turnover and cost in the next run reflect this).
  - So: **alpha_generator** turns train returns into alpha models; **pipeline** turns alpha models + current weights into new weights and cost. This gives a realistic, out-of-sample backtest with rebalancing and execution cost.

**Protocols**: Expects `ResearchPipelineLike` (e.g. `ResearchPipeline`) and `AlphaGeneratorLike` (callable: train_returns DataFrame → AlphaModelMap). Exposed in `ResearchExtensions.walk_forward` and `ai_qre.backtest.__init__`.

---

### `__init__.py`

Exports: `Backtester`, `PortfolioEnv`, `build_state`, `default_reward_fn`, `VectorizedBacktestResult`, `VectorizedResearchHarness`, `WalkForwardBacktester`, `WalkForwardResult`.
