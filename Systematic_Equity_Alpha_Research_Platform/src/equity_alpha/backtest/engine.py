"""
Backtest engine for the Systematic Equity Alpha Research Platform.

Given:
- a panel of alpha scores (predicted next-day returns),
- a portfolio construction rule (long/short),
- realised returns of the underlying assets,

we simulate the P&L of the strategy and compute standard performance
metrics (annualised return, vol, Sharpe, max drawdown, turnover).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from equity_alpha.data.data_loader import (
    load_price_panel,
    compute_log_returns,
)
from equity_alpha.models.linear_factor_model import (
    LinearFactorModel,
    LinearFactorModelConfig,
)
from equity_alpha.strategy.portfolio_construction import (
    LongShortConfig,
    build_long_short_weights,
)


@dataclass
class BacktestConfig:
    tickers: List[str]
    start: str
    end: str
    horizon: int = 1          # prediction horizon (days)
    long_quantile: float = 0.4
    short_quantile: float = 0.4
    gross_target: float = 1.0


# ----------------------------------------------------------------------
# Core backtest utilities
# ----------------------------------------------------------------------

def compute_portfolio_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute portfolio returns from weights and asset returns.

    Convention:
    - weights at date t are *decisions taken at the close of t*,
    - executed on returns from t+1.

    Implementation:
    - we shift weights by 1 day so that:
        r_p(t) = sum_i w_{t-1,i} * r_{t,i}

    Parameters
    ----------
    weights : DataFrame
        Portfolio weights (Date x Ticker).
    asset_returns : DataFrame
        Asset returns (Date x Ticker), here daily log-returns.

    Returns
    -------
    Series
        Portfolio returns (aligned on the Date index of asset_returns).
    """
    # Align index and columns
    common_dates = weights.index.intersection(asset_returns.index)
    common_cols = weights.columns.intersection(asset_returns.columns)

    w = weights.loc[common_dates, common_cols].copy()
    r = asset_returns.loc[common_dates, common_cols].copy()

    # Shift weights by 1 day to avoid look-ahead
    w_shifted = w.shift(1)

    # Elementwise multiply and sum across tickers
    port_rets = (w_shifted * r).sum(axis=1)

    # First day has NaN because we have no previous weights
    port_rets = port_rets.dropna()

    return port_rets


def compute_performance_stats(
    port_rets: pd.Series,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    Compute standard performance statistics from a series of portfolio
    returns (arithmetic, not log).

    Parameters
    ----------
    port_rets : Series
        Daily portfolio returns (can be small, e.g. ~0.001).
    trading_days : int, default 252
        Number of trading days per year.

    Returns
    -------
    dict
        Keys: 'ann_return', 'ann_vol', 'sharpe', 'max_drawdown'
    """
    port_rets = port_rets.dropna()
    if len(port_rets) < 2:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    # Cumulative equity curve
    equity = (1.0 + port_rets).cumprod()

    # Total return
    total_return = equity.iloc[-1] - 1.0

    n_days = len(port_rets)
    ann_return = (1.0 + total_return) ** (trading_days / n_days) - 1.0

    # Annualised volatility
    daily_vol = port_rets.std(ddof=0)
    ann_vol = daily_vol * np.sqrt(trading_days)

    sharpe = np.nan
    if ann_vol > 0 and not np.isnan(ann_vol):
        sharpe = ann_return / ann_vol

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = drawdown.min()

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def compute_turnover(
    weights: pd.DataFrame,
) -> float:
    """
    Compute average daily turnover as:

        0.5 * sum_i |w_{t,i} - w_{t-1,i}|

    averaged over t, ignoring the first date.
    """
    w = weights.sort_index()
    dw = w.diff().abs()
    # 0.5 * sum |Δw_i|
    daily_turnover = 0.5 * dw.sum(axis=1)
    # ignore first NaN
    daily_turnover = daily_turnover.dropna()
    if len(daily_turnover) == 0:
        return np.nan
    return float(daily_turnover.mean())


# ----------------------------------------------------------------------
# End-to-end backtest
# ----------------------------------------------------------------------

def run_backtest(cfg: BacktestConfig) -> Dict[str, object]:
    """
    End-to-end backtest:
    - fit linear factor model on [start, end]
    - compute scores on [start, end]
    - build long/short weights from scores
    - compute portfolio returns from realised asset returns
    - compute performance statistics

    Returns a dict with:
    - 'scores'   : DataFrame (Date x Ticker)
    - 'weights'  : DataFrame (Date x Ticker)
    - 'port_rets': Series
    - 'equity'   : Series (cumulative equity curve)
    - 'stats'    : dict of performance metrics
    """
    universe = cfg.tickers

    # 1) Fit the linear factor model
    model_cfg = LinearFactorModelConfig(
        tickers=universe,
        start=cfg.start,
        end=cfg.end,
        horizon=cfg.horizon,
    )

    model = LinearFactorModel()
    model.fit_from_raw(model_cfg)

    # 2) Compute scores over the full backtest window
    scores = model.predict_scores(
        tickers=universe,
        start=cfg.start,
        end=cfg.end,
    )

    # 3) Portfolio construction
    ls_cfg = LongShortConfig(
        long_quantile=cfg.long_quantile,
        short_quantile=cfg.short_quantile,
        gross_target=cfg.gross_target,
        min_names=2,
    )
    weights = build_long_short_weights(scores, ls_cfg)

    # 4) Realised returns of the universe
    prices = load_price_panel(
        tickers=universe,
        field="Adj Close",
        start=cfg.start,
        end=cfg.end,
    )
    asset_rets = compute_log_returns(prices, horizon=1)
    # convert log-returns to simple returns approx: r ≈ exp(log_r) - 1
    simple_asset_rets = np.expm1(asset_rets)

    # 5) Portfolio returns
    port_rets = compute_portfolio_returns(weights, simple_asset_rets)
    equity = (1.0 + port_rets).cumprod()

    # 6) Performance stats & turnover
    stats = compute_performance_stats(port_rets)
    stats["turnover"] = compute_turnover(weights)

    return {
        "scores": scores,
        "weights": weights,
        "port_rets": port_rets,
        "equity": equity,
        "stats": stats,
    }


# ----------------------------------------------------------------------
# Manual test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small end-to-end test on a tiny US universe.
    """

    universe = ["AAPL", "MSFT", "SPY", "META", "GOOGL"]
    cfg_bt = BacktestConfig(
        tickers=universe,
        start="2015-01-01",
        end="2024-01-01",
        horizon=1,
        long_quantile=0.4,
        short_quantile=0.4,
        gross_target=1.0,
    )

    res = run_backtest(cfg_bt)

    equity = res["equity"]
    stats = res["stats"]

    print("\n[backtest] Performance stats:")
    for k, v in stats.items():
        print(f"  {k:12s} = {v: .4f}")

    print("\n[backtest] Equity curve (last 10 points):")
    print(equity.tail(10))
