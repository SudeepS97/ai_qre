"""RL-friendly portfolio environment: reset() -> state, step(action) -> next_state, reward, done, info."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ai_qre.types import TradeVector, WeightVector


def default_reward_fn(
    pnl: float,
    cost: float,
    _state: dict[str, Any],
    _action: WeightVector,
    _info: dict[str, Any],
) -> float:
    """Reward = PnL net of trading costs."""
    return float(pnl - cost)


def build_state(
    positions: WeightVector,
    returns_window: np.ndarray,
    date_idx: int,
    tickers: list[str],
) -> dict[str, Any]:
    """Build standard state dict for RL: positions, returns window, date index, tickers."""
    return {
        "positions": dict(positions),
        "returns_window": np.asarray(returns_window, dtype=float) + 0.0,
        "date_idx": date_idx,
        "tickers": list(tickers),
    }


class PortfolioEnv:
    """
    Environment API for external RL agents: reset() and step(action_weights).
    State includes positions and a window of returns; reward is PnL minus costs by default.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        tickers: list[str],
        cost_fn: Callable[[TradeVector], float],
        state_window: int = 21,
        reward_fn: Callable[
            [float, float, dict[str, Any], WeightVector, dict[str, Any]],
            float,
        ] = default_reward_fn,
    ) -> None:
        if state_window < 1:
            raise ValueError("state_window must be >= 1")
        self.returns: pd.DataFrame = returns.reindex(columns=tickers).fillna(
            0.0
        )
        self.tickers: list[str] = list(tickers)
        self.cost_fn = cost_fn
        self.state_window = state_window
        self.reward_fn = reward_fn
        self._idx = 0
        self._positions: WeightVector = {t: 0.0 for t in self.tickers}
        self._n: int = len(self.tickers)
        self._T: int = len(self.returns)
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to start; return initial state and info."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._idx = self.state_window
        self._positions = {t: 0.0 for t in self.tickers}
        if self._idx >= self._T:
            state = build_state(
                self._positions,
                self.returns.iloc[: self.state_window].to_numpy(dtype=float),
                0,
                self.tickers,
            )
            return state, {"terminated": True, "truncated": True}
        window = self.returns.iloc[
            self._idx - self.state_window : self._idx
        ].to_numpy(dtype=float)
        state = build_state(self._positions, window, self._idx, self.tickers)
        return state, {"terminated": False, "truncated": False}

    def step(
        self, action: WeightVector | Mapping[str, float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply action (target weights); return next_state, reward, terminated, truncated, info."""
        action_mapping: Mapping[str, float] = (
            dict(action) if isinstance(action, dict) else dict(action)
        )
        trades: TradeVector = {
            t: float(action_mapping.get(t, 0.0) - self._positions.get(t, 0.0))
            for t in self.tickers
        }
        cost = self.cost_fn(trades)
        if self._idx >= self._T:
            state = build_state(
                self._positions,
                self.returns.iloc[-self.state_window :].to_numpy(dtype=float),
                self._idx,
                self.tickers,
            )
            return state, 0.0, True, True, {"pnl": 0.0, "cost": cost}
        row = self.returns.iloc[self._idx].to_numpy(dtype=float)
        weights_vec = np.asarray(
            [float(action_mapping.get(t, 0.0)) for t in self.tickers],
            dtype=float,
        )
        pnl = float((row * weights_vec).sum())
        reward = self.reward_fn(
            pnl, cost, {}, dict(action_mapping), {"pnl": pnl, "cost": cost}
        )
        self._positions = {
            t: float(action_mapping.get(t, 0.0)) for t in self.tickers
        }
        self._idx += 1
        terminated = False
        truncated = self._idx >= self._T
        if truncated:
            next_window = self.returns.iloc[-self.state_window :].to_numpy(
                dtype=float
            )
            next_idx = self._idx
        else:
            next_window = self.returns.iloc[
                self._idx - self.state_window : self._idx
            ].to_numpy(dtype=float)
            next_idx = self._idx
        next_state = build_state(
            self._positions, next_window, next_idx, self.tickers
        )
        info = {"pnl": pnl, "cost": cost}
        return next_state, reward, terminated, truncated, info
