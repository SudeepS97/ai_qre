# stress

This folder contains **stress testing**: simulating many possible future return paths for a given portfolio to summarize tail risk and drawdowns.

---

## Files

### `monte_carlo.py`

Single class: **`MonteCarloStress`**, which runs a **Monte Carlo simulation** assuming daily returns are multivariate normal with mean and covariance estimated from historical returns.

- **Constructor**: `MonteCarloStress(seed=None)`. `seed` fixes the random state for reproducibility (default 7).

- **`simulate(weights, returns, paths=1000, horizon=50) -> dict[str, float]`**
  - **Inputs**:
    - `weights`: mapping ticker → weight (e.g. from `build_portfolio`).
    - `returns`: DataFrame with tickers as columns; used to estimate mean vector and covariance matrix.
    - `paths`: number of simulated paths.
    - `horizon`: number of days per path.
  - **Process**:
    - Builds weight vector aligned to `returns.columns` (missing tickers get 0).
    - Estimates `mean_vector` = returns.mean(), `covariance` = returns.cov() (NumPy).
    - For each path: draws `horizon` days of returns from `multivariate_normal(mean_vector, covariance)`.
    - Daily PnL = simulated_returns @ weight_vector.
    - Equity curve = cumprod(1 + daily PnL).
    - For each path computes: terminal value (last point), worst drawdown (min of (equity / running_max - 1)), 95% VaR (0.95 quantile of losses), 95% CVaR (mean of losses >= that quantile).
  - **Output**: A dict of summary statistics over paths:
    - **Terminal**: `mean_terminal`, `median_terminal`, `p05_terminal`, `p95_terminal`, `worst_terminal`, `best_terminal`.
    - **Drawdown**: `mean_max_drawdown`, `worst_max_drawdown`.
    - **Risk**: `mean_var_95`, `mean_cvar_95`.

**Limitations**: Assumes i.i.d. multivariate normal returns; no regime shifts or fat tails. Use for quick comparative stress, not as a full risk model.

**Exposed in**: `ResearchExtensions.stress`.
