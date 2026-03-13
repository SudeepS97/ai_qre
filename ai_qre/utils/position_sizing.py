"""Convert portfolio weights (fractions of AUM) to share counts."""

from collections.abc import Mapping

import pandas as pd

from ai_qre.types import WeightVector


def weights_to_shares(
    weights: WeightVector,
    aum: float,
    prices: pd.Series | Mapping[str, float],
    *,
    round_shares: bool = False,
) -> dict[str, float]:
    """
    Convert weight vector and AUM to share counts using latest prices.

    - weight = fraction of AUM (e.g. 0.01 = 1%)
    - dollar_position = weight * aum
    - shares = dollar_position / price

    Returns dict[ticker, shares] where shares > 0 = long, shares < 0 = short.
    If round_shares is True, values are rounded to integers (for display/orders).
    """
    if isinstance(prices, pd.Series):
        price_map = prices.reindex(weights.keys()).fillna(0.0)
    else:
        price_map = {t: prices.get(t, 0.0) for t in weights}

    out: dict[str, float] = {}
    for ticker, weight in weights.items():
        price = float(price_map.get(ticker, 0.0))
        if price <= 0:
            out[ticker] = 0.0
            continue
        dollar_position = weight * aum
        n = dollar_position / price
        out[ticker] = round(n, 0) if round_shares else n
    return out


def shares_to_long_short(
    shares: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Split a shares dict into longs (positive) and shorts (absolute count).

    Returns (long_shares, short_shares) where each dict has ticker -> shares (positive).
    """
    longs: dict[str, float] = {}
    shorts: dict[str, float] = {}
    for ticker, n in shares.items():
        if n > 0:
            longs[ticker] = n
        elif n < 0:
            shorts[ticker] = -n
    return longs, shorts
