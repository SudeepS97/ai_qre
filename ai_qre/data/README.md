# data

This folder defines the **data interface** that the rest of the pipeline expects. There is no concrete implementation here—only the protocol your data source must implement.

---

## Files

### `provider.py`

Defines the **`MarketDataProvider`** protocol (structural subtyping; no inheritance required). Any class that implements these methods can be used as the `data` argument to `ResearchPipeline`, `LiquidityModel`, `ShrinkageCovariance`, `BarraLikeRiskModel`, etc.

- **`get_prices(tickers, start=None, end=None) -> pd.DataFrame`**

  - Returns a DataFrame with **index = dates** (typically business days) and **columns = tickers**.
  - Values = price per share.
  - `start` and `end` are optional datetime bounds; implementation can ignore them if not supported.

- **`get_returns(tickers, lookback=252) -> pd.DataFrame`**

  - Returns a DataFrame of **historical returns** (same index/columns as prices, or a subset).
  - The protocol provides a default implementation: call `get_prices(tickers)`, compute `pct_change()`, dropna, then take the last `lookback` rows.
  - So if you only implement `get_prices`, you get a simple returns implementation; you can override for different lookbacks or data sources.

- **`get_volumes(tickers) -> pd.DataFrame`**

  - Same index/columns convention as prices; values = **trading volume in shares** per day.
  - Used by `LiquidityModel` for average daily volume and ADDV.

- **`get_sectors(tickers) -> Mapping[str, str]`**

  - Returns a mapping **ticker → sector name** (e.g. `"AAPL" -> "Technology"`).
  - Used by `BarraLikeRiskModel` to build sector dummy exposures.

- **`get_market_caps(tickers) -> Mapping[str, float]`**
  - Returns **ticker → market cap** (float, typically in dollars).
  - Used by the Barra-like model for the size factor (e.g. log market cap).

**Usage**: Implement this protocol (e.g. wrap a database, CSV, or API) and pass the instance into `ResearchPipeline(data)`. See `example_usage.py` for a `MockData` class that implements the protocol with random data for testing.
