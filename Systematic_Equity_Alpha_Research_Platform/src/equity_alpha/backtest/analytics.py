"""
Analytics helpers for the equity alpha backtests.

This module is intentionally lightweight: it just provides
a nicely formatted summary of performance statistics and a
small helper to inspect weights / exposures.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def format_stats(stats: Dict[str, float]) -> str:
    """
    Format a dictionary of performance statistics as a readable string.
    Expected keys: ann_return, ann_vol, sharpe, max_drawdown, turnover.
    """
    parts = []
    parts.append(f"Annualised return : {stats.get('ann_return', np.nan): .4f}")
    parts.append(f"Annualised vol    : {stats.get('ann_vol', np.nan): .4f}")
    parts.append(f"Sharpe ratio      : {stats.get('sharpe', np.nan): .4f}")
    parts.append(f"Max drawdown      : {stats.get('max_drawdown', np.nan): .4f}")
    parts.append(f"Avg daily turnover: {stats.get('turnover', np.nan): .4f}")
    return "\n".join(parts)


def summarize_weights(weights: pd.DataFrame) -> str:
    """
    Simple diagnostics on the weight matrix:
    - average net exposure
    - average gross exposure
    - number of non-zero positions per day (avg)
    """
    w = weights.dropna(how="all").sort_index()
    if w.empty:
        return "No weights to summarize."

    net = w.sum(axis=1)
    gross = w.abs().sum(axis=1)
    nz = (w != 0.0).sum(axis=1)

    lines = [
        f"Average net exposure   : {net.mean(): .4f}",
        f"Average gross exposure : {gross.mean(): .4f}",
        f"Avg #non-zero positions: {nz.mean(): .2f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Tiny manual demo if needed
    import numpy as np

    stats_example = {
        "ann_return": 0.03,
        "ann_vol": 0.08,
        "sharpe": 0.4,
        "max_drawdown": -0.2,
        "turnover": 0.05,
    }
    print("[analytics] Example stats summary:")
    print(format_stats(stats_example))
