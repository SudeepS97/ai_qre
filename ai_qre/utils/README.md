# utils

This folder contains **shared utilities**: converting portfolio weights to share counts (for orders or reporting) and configuring structured logging.

---

## Files

### `position_sizing.py`

Converts **weights** (fractions of AUM) to **share counts** and splits them into long/short for reporting or order generation.

- **`weights_to_shares(weights, aum, prices, *, round_shares=False) -> dict[str, float]`**

  - **Inputs**:
    - `weights`: `WeightVector` (ticker → weight, e.g. 0.01 = 1% of AUM).
    - `aum`: total AUM in dollars.
    - `prices`: either a pd.Series (index = tickers) or a mapping ticker → price (e.g. last price).
  - **Logic**: For each ticker, dollar_position = weight × aum; shares = dollar_position / price. If price ≤ 0, returns 0 shares for that ticker.
  - **Output**: dict ticker → shares. **Sign**: positive = long, negative = short (same sign as weight).
  - If `round_shares=True`, values are rounded to integers (e.g. for order tickets).

- **`shares_to_long_short(shares: dict[str, float]) -> tuple[dict, dict]`**
  - **Input**: dict ticker → signed shares (positive = long, negative = short).
  - **Output**: `(long_shares, short_shares)` where each is a dict ticker → **positive** share count. Longs: tickers with positive shares; shorts: tickers with negative shares, with the value stored as the absolute count.
  - Useful for reporting “long book” and “short book” separately.

**Used by**: Callers after `build_portfolio` when they need actual share quantities (e.g. in `example_usage.py`).

---

### `logging.py`

Configures **structlog** and provides a logger getter so that pipeline and research code can log structured key-value pairs (and optionally JSON for production).

- **`configure_structlog(level=None, json=None, extra_processors=None)`**

  - Called automatically by `get_logger` on first use (or explicitly to set options). Runs only once unless `_CONFIGURED` is reset (e.g. by `init_structured_logging(force=True)`).
  - **Level**: from `level` or `LOG_LEVEL` env (e.g. `"INFO"`, `"DEBUG"`).
  - **Format**: JSON vs console from `json` or `LOG_JSON` env (default JSON if env not set).
  - Sets up stdlib `logging.basicConfig` and structlog processors (contextvars, level, timestamp, stack, exception).
  - Renderer: `JSONRenderer()` or `ConsoleRenderer()`.

- **`init_structured_logging(level=None, json=False, force=True)`**

  - Intended to be called at process start (e.g. in `main()`).
  - `force=True` resets the “already configured” flag so your `level`/`json` choices take effect even if something already called `get_logger`.
  - Default `json=False` gives human-readable console output; set `json=True` for JSON lines.

- **`get_logger(name=None, **bound_context) -> BoundLogger`\*\*

  - Ensures structlog is configured, then returns `structlog.get_logger(name)`. If `bound_context` is provided, returns `logger.bind(**bound_context)`.
  - Use for module-level loggers: `logger = get_logger(__name__)` then `logger.info("event", key=value)`.

- **`bind_context(**context)`** / **`clear_context()`\*\*

  - Set or clear structlog context vars for the current execution context (e.g. request id, run id). All subsequent log entries in that context will include these keys.

- **`add_standard_context(context)`**
  - Helper that returns `dict(context)`; mainly for consistency when building a standard set of fields to pass to `bind_context` or logs.

**Used by**: Application code and examples (e.g. `example_usage.py`) to get a logger and optionally init logging at startup.
