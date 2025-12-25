"""
Data loading utilities for the Systematic Equity Alpha Research Platform.

Goal:
- Download and cache daily equity data from yfinance.
- Provide clean price panels (aligned across tickers) ready for factor
  construction and backtesting.

This module intentionally stays simple and transparent: no hidden magic,
just a clear pipeline from raw prices to basic returns.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# We assume the project layout is:
#   project_root/
#     src/equity_alpha/data/data_loader.py
#     data/raw
#     data/clean
#
# From this file, project_root is two levels above (.. / ..).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"


def _ensure_directories() -> None:
    """Create data directories if they do not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Downloading utilities
# ---------------------------------------------------------------------------

def download_daily_prices(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    force: bool = False,
    auto_adjust: bool = False,
) -> Dict[str, Path]:
    """
    Download daily OHLCV data for a list of tickers from yfinance and cache
    them as Parquet files in data/raw.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols (e.g. ["AAPL", "MSFT", "SPY"]).
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str, optional
        End date in "YYYY-MM-DD" format. If None, yfinance uses "today".
    force : bool, default False
        If True, redownload even if a cached file already exists.
    auto_adjust : bool, default False
        If True, use yfinance's auto-adjusted prices (dividends / splits).
        Note: when auto_adjust=True, yfinance drops 'Adj Close', so in that
        case we only keep ['Open','High','Low','Close','Volume'] and
        treat 'Close' as adjusted.
    """
    _ensure_directories()
    cache_paths: Dict[str, Path] = {}

    for ticker in tickers:
        cache_file = RAW_DATA_DIR / f"{ticker}_daily.parquet"

        if cache_file.exists() and not force:
            print(f"[data] Using cached data for {ticker} -> {cache_file}")
            cache_paths[ticker] = cache_file
            continue

        print(f"[data] Downloading {ticker} from yfinance...")
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=auto_adjust,
        )

        if df.empty:
            raise ValueError(f"[data] No data returned for ticker {ticker}")

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        if auto_adjust:
            # yfinance already adjusted prices; no 'Adj Close' column.
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[cols]
        else:
            # keep explicit adjusted close
            cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            df = df[cols]

        df.to_parquet(cache_file)
        print(f"[data] Saved {ticker} data to {cache_file}")
        cache_paths[ticker] = cache_file

    return cache_paths



# ---------------------------------------------------------------------------
# Loading panels & basic transformations
# ---------------------------------------------------------------------------

def load_price_panel(
    tickers: List[str],
    field: str = "Adj Close",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a panel of prices for several tickers from the cached Parquet files,
    align them on a common Date index, and optionally clip to [start, end].

    Parameters
    ----------
    tickers : list of str
        Tickers to load.
    field : {"Open","High","Low","Close","Adj Close","Volume"}, default "Adj Close"
        Column from the raw data to extract.
    start : str, optional
        If provided, filter dates >= start ("YYYY-MM-DD").
    end : str, optional
        If provided, filter dates <= end ("YYYY-MM-DD").

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date, with one column per ticker.
        Missing values (e.g. holidays) are forward-filled then back-filled.
    """
    _ensure_directories()

    panel: Dict[str, pd.Series] = {}

    for ticker in tickers:
        cache_file = RAW_DATA_DIR / f"{ticker}_daily.parquet"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"[data] Cached file not found for {ticker}: {cache_file}. "
                f"Call download_daily_prices(...) first."
            )

        df = pd.read_parquet(cache_file)
        if field not in df.columns:
            raise ValueError(f"[data] Field '{field}' not found in {cache_file}")

        s = df[field].copy()
        s.name = ticker
        panel[ticker] = s

    prices = pd.concat(panel.values(), axis=1).sort_index()

    if start is not None:
        prices = prices.loc[prices.index >= pd.to_datetime(start)]
    if end is not None:
        prices = prices.loc[prices.index <= pd.to_datetime(end)]

    # Basic cleaning: forward-fill then back-fill missing values
    prices = prices.ffill().bfill()

    return prices


def compute_log_returns(
    prices: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Compute log-returns over a given horizon from a price panel.

    Parameters
    ----------
    prices : pd.DataFrame
        Price panel (Date index, one column per ticker).
    horizon : int, default 1
        Return horizon in days. horizon=1 gives daily log-returns,
        horizon=5 gives approximately weekly returns, etc.

    Returns
    -------
    pd.DataFrame
        Panel of log-returns aligned with the price index (first rows NaN).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    # log-return r_t = log(P_t / P_{t-horizon})
    log_p = np.log(prices)
    rets = log_p.diff(periods=horizon)

    return rets


# ---------------------------------------------------------------------------
# Example usage (manual test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small manual test:
    - download a few tickers if needed
    - build a price panel
    - compute daily log-returns
    """

    example_tickers = ["AAPL", "MSFT", "SPY"]

    download_daily_prices(
        example_tickers,
        start="2015-01-01",
        end="2024-01-01",
        force=False,
    )

    prices = load_price_panel(example_tickers, field="Adj Close")
    rets = compute_log_returns(prices, horizon=1)

    print("[data] Price panel head:")
    print(prices.head())
    print("\n[data] Return panel head:")
    print(rets.head())
