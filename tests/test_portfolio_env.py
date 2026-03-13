"""Tests for RL portfolio environment."""

import numpy as np
import pandas as pd

from ai_qre.backtest.portfolio_env import (
    PortfolioEnv,
    build_state,
    default_reward_fn,
)
from ai_qre.types import TradeVector


def test_build_state() -> None:
    positions = {"A": 0.1, "B": -0.1}
    window = np.array([[0.01, -0.02], [-0.01, 0.02]], dtype=float)
    state = build_state(positions, window, 5, ["A", "B"])
    assert state["positions"] == positions
    assert state["date_idx"] == 5
    assert state["tickers"] == ["A", "B"]
    assert state["returns_window"].shape == (2, 2)


def test_default_reward_fn() -> None:
    r = default_reward_fn(0.02, 0.001, {}, {}, {})
    assert abs(r - 0.019) < 1e-6


def test_portfolio_env_reset_step() -> None:
    returns = pd.DataFrame(
        [[0.01, -0.01], [0.02, 0.0], [-0.01, 0.02], [0.0, 0.01]],
        columns=["A", "B"],
    )

    def cost_fn(trades: TradeVector) -> float:
        return 0.001 * sum(abs(v) for v in trades.values())

    env = PortfolioEnv(returns, ["A", "B"], cost_fn, state_window=2)
    state, info = env.reset()
    assert "positions" in state and "returns_window" in state
    assert info["terminated"] is False or info["truncated"] is True
    action = {"A": 0.5, "B": -0.5}
    next_state, reward, term, trunc, step_info = env.step(action)
    assert "pnl" in step_info and "cost" in step_info
    assert next_state["positions"] == action
