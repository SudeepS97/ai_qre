"""Black-Litterman: blend prior expected returns with investor views."""

from collections.abc import Mapping, Sequence

import numpy as np

from ai_qre.types import AlphaVector


def _solve_symmetric_positive_definite(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Solve A @ x = b for x (Cholesky, no np.linalg). A must be symmetric positive definite."""
    n = b.shape[0]
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = float(a[i, j])
            for p in range(j):
                s -= L[i, p] * L[j, p]
            if i == j:
                L[i, j] = s**0.5
            else:
                L[i, j] = s / L[j, j]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        s = float(b[i])
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = float(y[i])
        for j in range(i + 1, n):
            s -= L[j, i] * x[j]
        x[i] = s / L[i, i]
    return x


def _views_to_pq(
    tickers: list[str],
    views: Sequence[tuple[str, Sequence[tuple[str, float]], float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build P (k x n) and Q (k,) from views. Each view is (label, [(ticker, coef), ...], return)."""
    ticker_index = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    k = len(views)
    P = np.zeros((k, n), dtype=float)
    Q = np.zeros(k, dtype=float)
    for i, (_label, pairs, ret) in enumerate(views):
        Q[i] = float(ret)
        for ticker, coef in pairs:
            if ticker in ticker_index:
                P[i, ticker_index[ticker]] = float(coef)
    return P, Q


def posterior_expected_returns(
    tickers: Sequence[str],
    prior_alphas: Mapping[str, float],
    cov_matrix: np.ndarray,
    views: Sequence[tuple[str, Sequence[tuple[str, float]], float]],
    tau: float = 0.025,
    omega_scale: float = 1.0,
) -> AlphaVector:
    """Black-Litterman posterior expected returns. Prior = prior_alphas; views blend in."""
    tickers_list = list(tickers)
    if not tickers_list or not views:
        return dict(prior_alphas)

    n = len(tickers_list)
    pi = np.asarray(
        [float(prior_alphas.get(t, 0.0)) for t in tickers_list],
        dtype=float,
    )
    Sigma = np.asarray(cov_matrix, dtype=float)
    if Sigma.shape != (n, n):
        return {t: float(prior_alphas.get(t, 0.0)) for t in tickers_list}

    P, Q = _views_to_pq(tickers_list, views)
    k = P.shape[0]
    if k == 0:
        return {t: float(pi[i]) for i, t in enumerate(tickers_list)}

    tau_Sigma = tau * Sigma
    P_tau_Sigma_P = P @ tau_Sigma @ P.T
    Omega = omega_scale * np.eye(k, dtype=float)
    M = P_tau_Sigma_P + Omega
    diff = Q - P @ pi
    M_sym = (M + M.T) / 2.0
    M_sym += 1e-12 * np.eye(k, dtype=float)
    try:
        x = _solve_symmetric_positive_definite(M_sym, diff)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        return {t: float(pi[i]) for i, t in enumerate(tickers_list)}
    posterior = pi + tau_Sigma @ P.T @ x
    return {
        ticker: float(posterior[i]) for i, ticker in enumerate(tickers_list)
    }
