# Optimization capabilities

This document summarizes what is **implemented** vs **not implemented** in the portfolio optimization stack.

---

## Implemented

- **Mean–variance**: Default objective (alpha · w − risk · Σ − turnover − factor penalty). `objective_type="mean_variance"`.
- **Global minimum variance (GMV)**: `objective_type="gmv"`.
- **CVaR**: Scenario-based CVaR objective. `objective_type="cvar"` (requires covariance provider with `.data` for scenario returns).
- **Tracking error**: Benchmark-relative. `objective_type="tracking_error"` with `benchmark_weights`.
- **Robust mean–variance**: Uncertainty set (box/ellipsoid). `objective_type="robust_mv"` with `uncertainty_radius`, `uncertainty_type`.
- **Black–Litterman**: Posterior expected returns from views; pipeline applies before optimizer when `use_black_litterman` and `bl_views` are set.
- **Resampled efficiency (Michaud)**: Bootstrap resampling of (μ, Σ); average weights over simulations. `use_resampled_efficiency`, `resampled_simulations`, `resampled_seed`.
- **Multi-period (MPC)**: First-period weights from a multi-period problem. `build_portfolio_mpc` and `solve_mpc_first_period`; WalkForwardConfig has `use_mpc`, `mpc_horizon`, `mpc_discount`.
- **Transaction cost in objective**: Per-asset quadratic trading cost in the optimizer objective. `use_trading_cost_in_objective`, `trading_cost_impact`; pipeline uses `LiquidityModel.trading_cost_impact_diag`.
- **Constraints**: Net target, gross limit, per-asset bounds, capacity-based caps, hard factor/sector neutrality, optional turnover limit, max names.

---

## Not implemented (examples)

- **Risk-parity**: No risk-parity or equal-risk-contribution objective.
- **Max-Sharpe as a single objective**: No direct max-Sharpe formulation (can be approximated via mean–variance with scaled alpha).
- **Full RL in pipeline**: No built-in RL agent; `PortfolioEnv` is provided for external RL use.
- **Other**: Downside risk (e.g. semivariance) as primary objective, custom factor models beyond Barra-like set, alternative solver backends beyond what CVXPY supports.

Data interface is protocol-only; you implement or adapt your data source. Solver is configurable (e.g. OSQP).
