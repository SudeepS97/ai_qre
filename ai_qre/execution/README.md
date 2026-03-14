# execution

This folder contains the **execution cost model**: how much it “costs” to trade a given size. Used to estimate the cost of a set of trades (e.g. the output of the optimizer) in dollars or as a drag on PnL.

---

## Files

### `simulator.py`

Single class: **`ExecutionSimulator`**, which models cost per trade as **linear (spread) + quadratic (impact)** in trade size.

- **Constructor**: `ExecutionSimulator(spread=0.0005, impact=0.1)`.

  - `spread`: cost per unit of notional (e.g. 5 bps).
  - `impact`: coefficient on the square of trade size (captures market impact; larger trades cost more per unit).

- **`cost(trade: float) -> float`**

  - For a single trade (signed: positive = buy, negative = sell), **size = |trade|** (e.g. as a weight or notional fraction).
  - **Cost = size × spread + impact × size².**
  - Returns a non-negative float. Typically you pass **weight** (fraction of portfolio) so cost is in “weight units”; to convert to dollars, multiply by AUM.

- **`portfolio_cost(trades: Mapping[str, float]) -> float`**
  - `trades`: ticker → signed trade (e.g. change in weight).
  - Sums `cost(trade)` over all trades.
  - Used by `ResearchPipeline.build_portfolio` to return the third element of the tuple `(weights, trades, cost)`.

**Note**: The `ExecutionConfig` dataclass in `ai_qre.config` has `spread_cost` and `impact_coeff` for documentation/defaults, but `ExecutionSimulator` does not read that dataclass; you set spread and impact via the constructor. The pipeline constructs `ExecutionSimulator()` with no arguments (so default 5 bps and 0.1).

**Usage**: The pipeline calls `exec_sim.portfolio_cost(trades)` after the optimizer to report cost. Backtests (e.g. walk-forward) can subtract this cost from the first day’s PnL when rebalancing.
