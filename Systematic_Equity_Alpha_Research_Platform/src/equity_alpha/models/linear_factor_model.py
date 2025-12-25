"""
Linear factor model for cross-sectional equity returns.

We take a small set of z-scored factors (momentum, volatility, liquidity)
and fit a pooled cross-sectional linear model:

    r_{i,t+1} ≈ beta_0 + beta^T f_{i,t}

where:
- i = ticker
- t = date
- f_{i,t} = vector of factors for stock i at date t
- r_{i,t+1} = next-day log-return

This is intentionally simple and global (one set of betas for the whole
sample). The goal is to produce a reasonable cross-sectional alpha score,
not to perfectly match academic factor models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from equity_alpha.data.data_loader import (
    load_price_panel,
    compute_log_returns,
    CLEAN_DATA_DIR,
)
from equity_alpha.data.factors import build_factors


# Factors we will use as predictors (all z-scored)
FACTOR_NAMES: List[str] = [
    "z_mom_12m_excl_1m",
    "z_mom_1m",
    "z_vol_20d",
    "z_vol_60d",
    "z_liq_20d_dv",
]


@dataclass
class LinearFactorModelConfig:
    tickers: List[str]
    start: str
    end: str
    horizon: int = 1  # prediction horizon for returns (in days)


class LinearFactorModel:
    """
    Simple linear cross-sectional factor model using pooled OLS.

    Usage:
        - call `fit_from_raw` with a config
        - call `predict_scores` to get alpha scores per date/ticker
    """

    def __init__(self, factor_names: List[str] | None = None):
        self.factor_names = factor_names or FACTOR_NAMES
        self.coef_: np.ndarray | None = None  # shape (n_factors + 1,)
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_factor_matrices(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load factor panels from disk for the given tickers and date range.

        Returns
        -------
        dict
            factor_name -> DataFrame (Date index, columns = tickers)
        """
        factors: Dict[str, pd.DataFrame] = {}
        for name in self.factor_names:
            path = CLEAN_DATA_DIR / f"{name}.parquet"
            df = pd.read_parquet(path)
            # Align on dates and tickers
            df = df.loc[
                (df.index >= pd.to_datetime(start))
                & (df.index <= pd.to_datetime(end))
            ]
            df = df.reindex(columns=tickers)
            factors[name] = df
        return factors

    def _build_regression_dataset(
        self,
        tickers: List[str],
        start: str,
        end: str,
        horizon: int,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Build the pooled cross-sectional regression dataset.

        Returns
        -------
        df_obs : DataFrame
            A long-format DataFrame with columns:
            ['Date', 'Ticker'] + factor_names + ['target']
        X : np.ndarray
            Design matrix with shape (n_obs, n_factors + 1) including intercept.
        y : np.ndarray
            Target vector with shape (n_obs,).
        """
        # 1) Load prices and compute returns
        prices = load_price_panel(tickers, field="Adj Close", start=start, end=end)
        rets = compute_log_returns(prices, horizon=horizon)
        # target = next-horizon return -> shift backwards
        target_panel = rets.shift(-horizon)

        # 2) Load factor matrices
        factor_mats = self._load_factor_matrices(tickers, start, end)

        rows = []
        for date in prices.index:
            for ticker in tickers:
                # target return at date (i.e. return from date -> date+horizon)
                y_val = target_panel.at[date, ticker] if ticker in target_panel.columns else np.nan
                if pd.isna(y_val):
                    continue

                # collect factor values
                f_vals = []
                nan_flag = False
                for fname in self.factor_names:
                    df_f = factor_mats[fname]
                    try:
                        val = df_f.at[date, ticker]
                    except KeyError:
                        val = np.nan
                    if pd.isna(val):
                        nan_flag = True
                        break
                    f_vals.append(val)

                if nan_flag:
                    continue

                rows.append([date, ticker] + f_vals + [y_val])

        if not rows:
            raise ValueError("No valid observations for regression dataset.")

        cols = ["Date", "Ticker"] + self.factor_names + ["target"]
        df_obs = pd.DataFrame(rows, columns=cols)

        # Build design matrix X (add intercept) and target y
        X_factors = df_obs[self.factor_names].to_numpy(dtype=float)
        intercept = np.ones((X_factors.shape[0], 1), dtype=float)
        X = np.hstack([intercept, X_factors])
        y = df_obs["target"].to_numpy(dtype=float)

        return df_obs, X, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_from_raw(self, cfg: LinearFactorModelConfig) -> pd.DataFrame:
        """
        End-to-end:
        - ensure factors exist (calls build_factors)
        - build regression dataset
        - fit pooled OLS
        Returns the DataFrame of observations (for inspection / diagnostics).
        """
        # Ensure factors are available on disk (no re-download if already there)
        build_factors(
            tickers=cfg.tickers,
            start=cfg.start,
            end=cfg.end,
            force_download=False,
        )

        df_obs, X, y = self._build_regression_dataset(
            tickers=cfg.tickers,
            start=cfg.start,
            end=cfg.end,
            horizon=cfg.horizon,
        )

        # OLS via lstsq: min ||X beta - y||^2
        beta, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
        self.coef_ = beta
        self._fitted = True

        # Simple printout for now
        print("\n[linear_model] Fitted pooled OLS model:")
        print(f"  n_obs     = {X.shape[0]}")
        print(f"  n_factors = {len(self.factor_names)}")
        print("\n  Coefficients (beta):")
        print(f"    intercept: {beta[0]: .4e}")
        for name, b in zip(self.factor_names, beta[1:]):
            print(f"    {name:>15}: {b: .4e}")

        return df_obs

    def predict_scores(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Use the fitted linear model to produce alpha scores (predicted
        next-day log-returns) for each date and ticker in the given
        range.

        Returns
        -------
        DataFrame
            Index = Date, columns = tickers, values = predicted returns.
        """
        if not self._fitted or self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit_from_raw(...) first.")

        factor_mats = self._load_factor_matrices(tickers, start, end)

        # We'll build scores date x ticker
        all_dates = None
        for df in factor_mats.values():
            all_dates = df.index if all_dates is None else all_dates.intersection(df.index)

        if all_dates is None:
            raise ValueError("No dates available in factor matrices.")

        all_dates = sorted(all_dates)
        scores = pd.DataFrame(index=all_dates, columns=tickers, dtype=float)

        beta = self.coef_
        beta0 = beta[0]
        beta_f = beta[1:]

        for date in all_dates:
            for ticker in tickers:
                f_vals = []
                nan_flag = False
                for fname in self.factor_names:
                    df_f = factor_mats[fname]
                    try:
                        val = df_f.at[date, ticker]
                    except KeyError:
                        val = np.nan
                    if pd.isna(val):
                        nan_flag = True
                        break
                    f_vals.append(val)

                if nan_flag:
                    scores.at[date, ticker] = np.nan
                    continue

                f_vec = np.array(f_vals, dtype=float)
                scores.at[date, ticker] = beta0 + float(np.dot(beta_f, f_vec))

        return scores


# ----------------------------------------------------------------------
# Manual test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small manual test:
    - build factors for a tiny universe
    - fit the linear factor model
    - compute alpha scores on the last 60 days
    """

    universe = ["AAPL", "MSFT", "SPY", "META", "GOOGL"]
    cfg = LinearFactorModelConfig(
        tickers=universe,
        start="2015-01-01",
        end="2024-01-01",
        horizon=1,
    )

    model = LinearFactorModel()
    df_obs = model.fit_from_raw(cfg)

    # Take last ~60 days of sample for score illustration
    last_start = "2023-10-01"
    last_end = "2024-01-01"
    scores = model.predict_scores(universe, start=last_start, end=last_end)

    print("\n[linear_model] Score sample (last few dates):")
    print(scores.tail())

    # Example: last date ranking
    last_date = scores.dropna(how="all").index.max()
    last_scores = scores.loc[last_date].dropna().sort_values(ascending=False)
    print(f"\n[linear_model] Ranking on {last_date.date()}:")
    for ticker, sc in last_scores.items():
        print(f"  {ticker}: {sc: .4e}")
