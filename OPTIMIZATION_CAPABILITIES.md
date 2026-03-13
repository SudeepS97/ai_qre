## Optimization and Risk Capabilities in `ai_qre`

This document summarizes how the current `ai_qre` codebase maps to the portfolio optimization concepts described in the “Engineering of Alpha” optimization report.

### Implemented Concepts

- **Single-period mean–variance-style optimizer**

  - Implemented by `PortfolioOptimizer` in `ai_qre/portfolio/optimizer.py`.
  - Objective: maximize alpha subject to a quadratic risk penalty (`risk_aversion`) and an L1 turnover penalty, using a covariance matrix from `ShrinkageCovariance`.
  - Supports:
    - Gross and net exposure constraints.
    - Per-asset max position constraints.
    - Optional hard factor- and sector-neutrality constraints via factor exposures.
    - Optional capacity-driven per-asset max-weight limits.

- **Covariance estimation with shrinkage**

  - Implemented by `ShrinkageCovariance` in `ai_qre/risk/covariance.py`.
  - Uses sample covariance of historical returns with diagonal shrinkage, consistent with shrinkage techniques discussed in modern mean–variance implementations.

- **Factor risk model (Barra-like)**

  - Implemented by `BarraLikeRiskModel` and `FactorRiskSnapshot` in `ai_qre/risk/barra_model.py`.
  - Supports:
    - Market beta, size (log market cap), momentum, and sector dummy factors.
    - Construction of factor covariance, idiosyncratic variances, and full asset covariance.
  - Exposed via `ResearchExtensions` and used by `ResearchPipeline` for factor exposures and neutralization.

- **Alpha combination and transforms**

  - Implemented in `ai_qre/alpha/transforms.py`:
    - `AlphaBlender` for combining multiple alpha models with weights.
    - `AlphaDecay` for half-life style decay of signals.
    - `shrink` for prior-mean shrinkage of alphas.
    - `orthogonalize` for QR-based orthogonalization of alpha matrices.
  - These correspond to the document’s discussion of alpha blending, shrinkage, and orthogonalization for robust signal integration.

- **Cross-sectional regression alpha model**

  - Implemented by `CrossSectionalAlphaModel` in `ai_qre/alpha/cross_sectional_regression.py`.
  - Provides cross-sectional OLS with ridge stabilization for estimating relationships between signals and future returns.

- **Capacity and liquidity modeling**

  - Implemented by `LiquidityModel` in `ai_qre/capacity/liquidity.py`.
  - Computes average daily dollar volume (ADDV) and derives per-asset max-weight limits based on AUM, participation caps, and days-to-liquidate.
  - This aligns with capacity and liquidity constraints (ADV-based limits) described in the report.

- **Execution cost model**

  - Implemented by `ExecutionSimulator` in `ai_qre/execution/simulator.py`.
  - Models per-trade costs as spread cost plus a quadratic impact term in trade size.
  - Used in `ResearchPipeline` and `WalkForwardBacktester` to deduct costs ex post at each rebalance.

- **Backtesting and walk-forward evaluation**

  - `Backtester` in `ai_qre/backtest/backtester.py` implements vectorized backtests for static weight vectors.
  - `WalkForwardBacktester` in `ai_qre/backtest/walk_forward.py` implements rolling train/test splits with periodic re-optimization, turnover tracking, and cost deductions.
  - These provide a myopic single-period-to-single-period evaluation framework.

- **Research orchestration**
  - `ResearchPipeline` in `ai_qre/research_pipeline.py` wires together data, alpha models, risk models, optimizer, liquidity limits, and execution cost simulation.
  - `ResearchExtensions` in `ai_qre/research_extensions.py` exposes a typed facade for advanced utilities (Barra-like risk, cross-sectional regression, walk-forward, vectorized harness, stress tests, distributed research, experiment tracking).

### Not Yet Implemented (Gaps vs. Document)

The following concepts from the optimization report are **not yet implemented** in `ai_qre` and are targets for future work:

- **Alternative objective families**

  - Global Minimum Variance (GMV) objective.
  - Explicit maximum Sharpe ratio / risk-budgeted objectives.
  - Benchmark-relative / tracking-error objectives and Information Ratio maximization.
  - Risk-parity / equal-risk-contribution portfolios.
  - Maximum diversification ratio portfolios.

- **Benchmark-relative and factor-relative control**

  - Explicit tracking-error minimization vs a benchmark portfolio.
  - Constraints and objectives defined in terms of deviations from benchmark factor exposures.

- **Downside risk measures**

  - CVaR / Expected Shortfall optimization.
  - Drawdown-aware objectives (CDaR, max drawdown constraints).

- **Robust optimization and resampled efficiency**

  - Box or ellipsoidal uncertainty sets around alphas or covariances.
  - Michaud-style resampled efficient frontier.

- **Black–Litterman framework**

  - Reverse-optimized equilibrium returns.
  - Integration of investor views (`P, Q, Omega`) into posterior return estimates.

- **Transaction-cost-aware objectives**

  - Direct inclusion of execution costs and market impact in the optimization objective (e.g., quadratic penalty in trades).

- **Multi-period optimization and MPC**

  - Explicit multi-period trajectory optimization over a horizon with dynamic constraints.
  - Model Predictive Control (MPC)-style rolling re-optimization that plans a path of trades.

- **Reinforcement learning integration**
  - Environment-style APIs (`reset`, `step`) for training RL agents on portfolio decisions.
  - RL-based policies that directly map state to portfolio weights or trades.

This document is intended as a reference point for ongoing development so that future PRs can close the gaps systematically while keeping the architecture cohesive.
