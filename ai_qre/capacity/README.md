# capacity

This folder contains **liquidity and capacity** logic: how much of each asset can be held or traded without exceeding participation and liquidation assumptions. Used to set per-asset max weight limits in the optimizer.

---

## Files

### `liquidity.py`

Single module: **`LiquidityModel`**, which uses average daily volume and config to compute maximum position sizes (as weights) and a capacity report.

- **Dependencies**: `MarketDataProvider` (for volumes and prices), `CapacityConfig` from `ai_qre.config`.

- **Constructor**: `LiquidityModel(data: MarketDataProvider, config: CapacityConfig | None = None)`. Uses default `CapacityConfig()` if not provided. Config fields: `adv_fraction`, `participation_cap`, `forecast_days_to_liquidate`, `min_weight_cap`.

- **`average_daily_dollar_volume(tickers) -> pd.Series`**

  - Gets volume (shares) and prices from the data provider.
  - ADV in shares = mean of daily volume per ticker.
  - ADDV = ADV × last available price.
  - Returns a Series indexed by ticker, name `"addv"`.

- **`max_weight_limits(tickers, aum: float) -> pd.Series`**

  - **Idea**: Cap each position so that, at a given participation rate and over a fixed number of days, you could liquidate that position without exceeding participation.
  - For each ticker: “liquidatable” dollar amount = ADDV × `participation_cap` × `forecast_days_to_liquidate`.
  - Max weight = liquidatable_dollars / AUM, then clipped below by `min_weight_cap`.
  - Raises if `aum <= 0`.
  - Returns Series indexed by ticker, name `"max_weight"`. The pipeline passes this as per-asset upper (and symmetric lower) bounds to the optimizer when `use_capacity_limits` is True.

- **`capacity_report(weights, aum: float) -> pd.DataFrame`**
  - Takes the current portfolio weights (dict or similar).
  - Computes `max_weight` per ticker via `max_weight_limits`.
  - Builds a DataFrame with columns: `abs_weight` (absolute weight), `max_weight`, and `capacity_usage` = abs_weight / max_weight (NaN when max_weight is 0).
  - Sorted by `capacity_usage` descending so the most capacity-constrained names appear first.
  - Useful for compliance and risk reporting.

**Used by**: `ResearchPipeline` when `portfolio_config.use_capacity_limits` is True; it calls `liquidity.max_weight_limits(tickers, aum)` and passes the result into `PortfolioOptimizer.solve(..., max_weight_by_asset=...)`.
