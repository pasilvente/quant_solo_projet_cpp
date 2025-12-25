"""
Portfolio construction from cross-sectional alpha scores.

Given a panel of alpha scores (predicted next-day returns) with shape
(Date x Ticker), we build a simple long/short equity portfolio:

- at each date, rank stocks by score,
- go long the top X%,
- go short the bottom X%,
- scale weights to be:
    * dollar-neutral (sum(weights) = 0)
    * with a target gross exposure (sum(|weights|) = gross_target)

This is intentionally simple but realistic enough for a first pass.
More sophisticated schemes (sector-neutrality, risk parity, etc.) can
be added later on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class LongShortConfig:
    long_quantile: float = 0.3    # fraction of names to go long
    short_quantile: float = 0.3   # fraction of names to go short
    gross_target: float = 1.0     # sum of absolute weights at each date
    min_names: int = 2            # minimum number of names in long AND short


def build_long_short_weights(
    scores: pd.DataFrame,
    cfg: LongShortConfig,
) -> pd.DataFrame:
    """
    Build a long/short portfolio from alpha scores.

    Parameters
    ----------
    scores : DataFrame
        Alpha scores with index = Date and columns = tickers.
        Higher scores mean higher expected return.
    cfg : LongShortConfig
        Configuration for quantiles and target gross exposure.

    Returns
    -------
    DataFrame
        Weights with the same index/columns as `scores`.
        Each row sums approximately to 0 (dollar-neutral) and has
        sum(|w|) ≈ gross_target (unless there are too few names).
    """
    if not 0 < cfg.long_quantile <= 1.0:
        raise ValueError("long_quantile must be in (0, 1].")
    if not 0 < cfg.short_quantile <= 1.0:
        raise ValueError("short_quantile must be in (0, 1].")

    weights = pd.DataFrame(index=scores.index, columns=scores.columns, dtype=float)
    weights[:] = 0.0

    for date, row in scores.iterrows():
        s = row.dropna()
        n = len(s)
        if n < 2 * cfg.min_names:
            # Not enough names to build a meaningful long/short book
            continue

        # Sort by score descending
        s_sorted = s.sort_values(ascending=False)

        # Determine how many names to take long/short
        n_long = max(cfg.min_names, int(np.floor(cfg.long_quantile * n)))
        n_short = max(cfg.min_names, int(np.floor(cfg.short_quantile * n)))

        if n_long + n_short > n:
            # In extreme small-universe cases, adjust
            n_long = max(cfg.min_names, n // 2)
            n_short = n - n_long

        long_names = s_sorted.index[:n_long]
        short_names = s_sorted.index[-n_short:]

        # Raw weights: equal-weight within long and short
        w = pd.Series(0.0, index=s.index)
        if n_long > 0:
            w[long_names] = 1.0 / n_long
        if n_short > 0:
            w[short_names] = -1.0 / n_short

        # At this stage, gross = sum(|w|), net may not be exactly 0
        gross = np.abs(w).sum()
        if gross == 0 or np.isnan(gross):
            continue

        # Rescale to target gross exposure
        w = w * (cfg.gross_target / gross)

        # Force small numerical zero on net exposure (optional)
        # net = w.sum()  # we could log it for diagnostics

        # Store into the weights DataFrame
        weights.loc[date, w.index] = w

    return weights


# ----------------------------------------------------------------------
# Manual test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small manual test:
    - fit the linear factor model on a small universe
    - compute scores on the last part of the sample
    - build long/short weights from these scores
    """

    from equity_alpha.models.linear_factor_model import (
        LinearFactorModel,
        LinearFactorModelConfig,
    )

    universe: List[str] = ["AAPL", "MSFT", "SPY", "META", "GOOGL"]

    cfg_model = LinearFactorModelConfig(
        tickers=universe,
        start="2015-01-01",
        end="2024-01-01",
        horizon=1,
    )

    model = LinearFactorModel()
    model.fit_from_raw(cfg_model)

    # Scores over a recent window
    score_start = "2023-10-01"
    score_end = "2024-01-01"
    scores = model.predict_scores(universe, start=score_start, end=score_end)

    ls_cfg = LongShortConfig(
        long_quantile=0.4,
        short_quantile=0.4,
        gross_target=1.0,
        min_names=2,
    )

    weights = build_long_short_weights(scores, ls_cfg)

    print("\n[portfolio] Sample of scores (last 5 dates):")
    print(scores.tail())

    print("\n[portfolio] Sample of weights (last 5 dates):")
    print(weights.tail())

    # Sanity check on the last date: net & gross exposure
    last_date = weights.dropna(how="all").index.max()
    w_last = weights.loc[last_date].dropna()
    print(f"\n[portfolio] Diagnostics on {last_date.date()}:")
    print(f"  net exposure   = {w_last.sum(): .4e}")
    print(f"  gross exposure = {np.abs(w_last).sum(): .4e}")
    print("  weights by name:")
    for name, w in w_last.items():
        print(f"    {name}: {w: .4e}")
