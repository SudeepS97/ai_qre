"""Tests for Black-Litterman posterior expected returns."""

import numpy as np

from ai_qre.portfolio.black_litterman import posterior_expected_returns


def test_posterior_returns_no_views_returns_prior() -> None:
    tickers = ["A", "B"]
    prior = {"A": 0.01, "B": 0.02}
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    out = posterior_expected_returns(tickers, prior, cov, [], tau=0.025)
    assert out == prior


def test_posterior_returns_with_absolute_view_shifts_alphas() -> None:
    tickers = ["A", "B"]
    prior = {"A": 0.01, "B": 0.02}
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    views = (("v1", (("A", 1.0),), 0.05),)
    out = posterior_expected_returns(
        tickers, prior, cov, views, tau=0.025, omega_scale=0.1
    )
    assert "A" in out and "B" in out
    assert out["A"] != prior["A"]
    assert abs(out["A"] - 0.05) < abs(prior["A"] - 0.05)


def test_posterior_returns_empty_tickers_returns_prior() -> None:
    prior = {"A": 0.01}
    cov = np.array([[0.04]], dtype=float)
    out = posterior_expected_returns(
        [], prior, cov, (("v", (("A", 1.0),), 0.05),)
    )
    assert out == prior
