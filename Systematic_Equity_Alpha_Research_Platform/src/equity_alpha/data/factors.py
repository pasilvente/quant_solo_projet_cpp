"""
Factor construction for the Systematic Equity Alpha Research Platform.

This module builds a small set of classic cross-sectional equity factors
from daily price and volume data:

- Momentum (12M-1M and 1M)
- Realised volatility (20d, 60d)
- Liquidity proxy (rolling dollar volume)

The goal is not to perfectly replicate academic definitions, but to have
clean, transparent implementations that are easy to extend.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from equity_alpha.data.data_loader import (
    download_daily_prices,
    load_price_panel,
    compute_log_returns,
    CLEAN_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _zscore_cross_section(df: pd.DataFrame, min_valid: int = 5) -> pd.DataFrame:
    """
    Cross-sectional z-score at each date.

    For each row (date), subtract the cross-sectional mean and divide by
    the cross-sectional standard deviation. Rows with fewer than
    `min_valid` non-NaN entries are left as NaN.
    """
    def _z(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if valid.size < min_valid:
            return pd.Series(np.nan, index=row.index)
        mu = valid.mean()
        sigma = valid.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(np.nan, index=row.index)
        return (row - mu) / sigma

    return df.apply(_z, axis=1)


def _momentum_log(
    prices: pd.DataFrame,
    lookback: int,
    lag: int = 0,
) -> pd.DataFrame:
    """
    Log-momentum over a given lookback, optionally skipping the most
    recent `lag` days.

    Example:
    - 12M-1M momentum ~ lookback=252, lag=21

    Definition:
        mom_t = log(P_{t-lag}) - log(P_{t-lag-lookback})
    """
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if lag < 0:
        raise ValueError("lag must be >= 0")

    log_p = np.log(prices)
    return log_p.shift(lag) - log_p.shift(lag + lookback)


def _realised_vol(
    returns: pd.DataFrame,
    window: int,
    annualise: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Rolling realised volatility of (log) returns over a given window.

    Parameters
    ----------
    returns : DataFrame
        Daily log-returns (one column per ticker).
    window : int
        Rolling window length in days.
    annualise : bool, default True
        If True, multiply by sqrt(trading_days).
    trading_days : int, default 252
        Number of trading days per year.

    Returns
    -------
    DataFrame
        Rolling volatility, same shape as `returns`.
    """
    if window <= 1:
        raise ValueError("window must be > 1")

    vol = returns.rolling(window=window).std(ddof=0)
    if annualise:
        vol = vol * np.sqrt(trading_days)
    return vol


def _rolling_dollar_volume(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Rolling average dollar volume: mean_t (Price_t * Volume_t) over a
    given window. Used as a simple liquidity proxy.
    """
    dollar_vol = prices * volumes
    return dollar_vol.rolling(window=window).mean()


# ---------------------------------------------------------------------------
# Main factor construction
# ---------------------------------------------------------------------------

def build_factors(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Build a panel of equity factors for a given ticker universe and date
    range. Also saves the result to data/clean as Parquet files.

    Factors implemented:
    - mom_12m_excl_1m : 12-month momentum excluding last month (approx.)
    - mom_1m          : 1-month momentum
    - vol_20d         : 20-day realised volatility (annualised)
    - vol_60d         : 60-day realised volatility (annualised)
    - liq_20d_dv      : 20-day average dollar volume
    - z_*             : cross-sectional z-scores of the above (per date)
    """
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure raw data is available
    download_daily_prices(
        tickers,
        start=start,
        end=end,
        force=force_download,
        auto_adjust=False,  # keep explicit 'Adj Close'
    )

    # Load prices and volumes
    prices = load_price_panel(tickers, field="Adj Close", start=start, end=end)
    volumes = load_price_panel(tickers, field="Volume", start=start, end=end)
    rets = compute_log_returns(prices, horizon=1)

    # Momentum factors
    mom_12m_excl_1m = _momentum_log(prices, lookback=252, lag=21)
    mom_1m = _momentum_log(prices, lookback=21, lag=0)

    # Volatility factors
    vol_20d = _realised_vol(rets, window=20, annualise=True)
    vol_60d = _realised_vol(rets, window=60, annualise=True)

    # Liquidity proxy
    liq_20d_dv = _rolling_dollar_volume(prices, volumes, window=20)

    # Base factors (bruts)
    base_factors: Dict[str, pd.DataFrame] = {
        "mom_12m_excl_1m": mom_12m_excl_1m,
        "mom_1m": mom_1m,
        "vol_20d": vol_20d,
        "vol_60d": vol_60d,
        "liq_20d_dv": liq_20d_dv,
    }

    # Cross-sectional z-scores (per date), dans un dict séparé
    z_factors: Dict[str, pd.DataFrame] = {}
    for name, df in base_factors.items():
        zname = f"z_{name}"
        z_factors[zname] = _zscore_cross_section(df)

    # Fusionner bruts + z-scores dans un seul dict
    factors: Dict[str, pd.DataFrame] = {}
    factors.update(base_factors)
    factors.update(z_factors)

    # Sauvegarde de chaque facteur + panel combiné
    for name, df in factors.items():
        out_path = CLEAN_DATA_DIR / f"{name}.parquet"
        df.to_parquet(out_path)
        print(f"[factors] Saved {name} to {out_path}")

    combined = pd.concat(
        {name: df for name, df in factors.items()},
        axis=1
    )
    combined_out = CLEAN_DATA_DIR / "factors_panel.parquet"
    combined.to_parquet(combined_out)
    print(f"[factors] Saved combined factor panel to {combined_out}")

    return factors



# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small manual test on a tiny universe.
    """

    universe = ["AAPL", "MSFT", "SPY", "META", "GOOGL"]

    factors = build_factors(
        tickers=universe,
        start="2015-01-01",
        end="2024-01-01",
        force_download=False,
    )

    print("\n[factors] Example: head of z_mom_12m_excl_1m:")
    print(factors["z_mom_12m_excl_1m"].head())
