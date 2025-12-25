"""
Machine-learning models for cross-sectional equity returns.

This module reuses the same factor set as the linear factor model and
benchmarks a few standard models:

- plain linear regression (as a baseline),
- Lasso (L1-regularised linear model),
- Random Forest regressor,
- XGBoost regressor,
- LightGBM regressor.

The goal here is not to claim huge predictive power – daily and even
weekly equity returns are extremely noisy – but to set up a clean,
reproducible ML pipeline:

- build a long-format dataset (Date, Ticker, features, target),
- split it into train / test based on time,
- train several models with reasonable hyperparameters,
- compare out-of-sample performance (R^2, MSE, information coefficient).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Optional: XGBoost and LightGBM
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from equity_alpha.data.data_loader import (
    load_price_panel,
    compute_log_returns,
    CLEAN_DATA_DIR,
)
from equity_alpha.data.factors import build_factors
from equity_alpha.models.linear_factor_model import FACTOR_NAMES


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLModelConfig:
    """
    Configuration for the ML backtest on cross-sectional returns.

    Parameters
    ----------
    tickers : list of str
        Universe of stocks.
    start, end : str
        Backtest window (YYYY-MM-DD).
    horizon : int
        Return horizon in days, e.g. 1 for daily, 5 for weekly.
    test_fraction : float
        Fraction of dates assigned to the test set (chronological split).
    random_state : int
        Random seed for models that use randomness.
    """
    tickers: List[str]
    start: str
    end: str
    horizon: int = 5
    test_fraction: float = 0.2
    random_state: int = 42


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _load_factor_matrices(
    factor_names: List[str],
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load factor panels from disk for the given tickers and date range.

    Returns
    -------
    dict
        factor_name -> DataFrame with Date index and ticker columns.
    """
    factors: Dict[str, pd.DataFrame] = {}
    for name in factor_names:
        path = CLEAN_DATA_DIR / f"{name}.parquet"
        df = pd.read_parquet(path)
        df = df.loc[
            (df.index >= pd.to_datetime(start))
            & (df.index <= pd.to_datetime(end))
        ]
        df = df.reindex(columns=tickers)
        factors[name] = df
    return factors


def build_ml_dataset(
    tickers: List[str],
    start: str,
    end: str,
    horizon: int,
    factor_names: List[str],
) -> pd.DataFrame:
    """
    Build a long-format dataset for ML.

    Each row corresponds to a (date, ticker) pair with:

        [Date, Ticker] + factor values at date t + target return over
        [t, t + horizon].

    Parameters
    ----------
    tickers : list of str
        Universe.
    start, end : str
        Date range.
    horizon : int
        Return horizon in days.
    factor_names : list of str
        Names of the factor columns to use as features.

    Returns
    -------
    DataFrame
        Columns: ["Date", "Ticker"] + factor_names + ["target"].
    """
    # 1) Prices and returns
    prices = load_price_panel(tickers, field="Adj Close", start=start, end=end)
    rets = compute_log_returns(prices, horizon=horizon)
    # target return from t to t+horizon
    target_panel = rets.shift(-horizon)

    # 2) Factors
    factor_mats = _load_factor_matrices(factor_names, tickers, start, end)

    rows = []
    for date in prices.index:
        for ticker in tickers:
            y_val = target_panel.at[date, ticker] if ticker in target_panel.columns else np.nan
            if pd.isna(y_val):
                continue

            f_vals = []
            nan_flag = False
            for fname in factor_names:
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
        raise ValueError("No valid observations for ML dataset.")

    cols = ["Date", "Ticker"] + factor_names + ["target"]
    df_obs = pd.DataFrame(rows, columns=cols)
    df_obs.sort_values(["Date", "Ticker"], inplace=True)
    df_obs.reset_index(drop=True, inplace=True)

    return df_obs


def train_test_split_by_time(
    df_obs: pd.DataFrame,
    test_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / test split.

    We split on unique dates:
    - early dates → train
    - most recent dates → test.

    This is closer to a realistic research setup than a random shuffle.
    """
    if not 0.0 < test_fraction < 0.9:
        raise ValueError("test_fraction should be in (0, 0.9).")

    unique_dates = df_obs["Date"].sort_values().unique()
    n_dates = len(unique_dates)
    n_test = max(1, int(np.floor(test_fraction * n_dates)))
    split_date = unique_dates[-n_test]  # first date in the test period

    df_train = df_obs[df_obs["Date"] < split_date].copy()
    df_test = df_obs[df_obs["Date"] >= split_date].copy()

    return df_train, df_test


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Fit a model and compute basic regression metrics on train and test sets.

    We deliberately focus on simple summary metrics:
    - R^2 (how much variance we explain),
    - MSE (scale of the error),
    - information coefficient (corr between predictions and realised returns).
    """
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    if np.std(y_test) > 0 and np.std(y_pred_test) > 0:
        ic_test = float(np.corrcoef(y_test, y_pred_test)[0, 1])
    else:
        ic_test = np.nan

    print(f"\n[ml] {name} results:")
    print(f"  Train R^2   = {train_r2:.4e}")
    print(f"  Test  R^2   = {test_r2:.4e}")
    print(f"  Train MSE   = {train_mse:.4e}")
    print(f"  Test  MSE   = {test_mse:.4e}")
    print(f"  Test IC     = {ic_test:.4e}")

    return {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "test_ic": ic_test,
    }


def run_ml_benchmark(cfg: MLModelConfig) -> Dict[str, Dict[str, float]]:
    """
    End-to-end ML benchmark on cross-sectional equity returns.

    Steps:
    - ensure factors exist on disk,
    - build the long-format dataset,
    - split into train/test by time,
    - train and evaluate several models:
        * LinearRegression
        * LassoCV
        * RandomForestRegressor
        * XGBRegressor (if available)
        * LGBMRegressor (if available)

    Returns
    -------
    dict
        model_name -> dict of metrics (train/test R^2, MSE, test IC).
    """
    # Make sure factor files exist (no re-download if already cached)
    build_factors(
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        force_download=False,
    )

    df_obs = build_ml_dataset(
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        horizon=cfg.horizon,
        factor_names=FACTOR_NAMES,
    )

    df_train, df_test = train_test_split_by_time(df_obs, cfg.test_fraction)

    X_train = df_train[FACTOR_NAMES].to_numpy(dtype=float)
    y_train = df_train["target"].to_numpy(dtype=float)
    X_test = df_test[FACTOR_NAMES].to_numpy(dtype=float)
    y_test = df_test["target"].to_numpy(dtype=float)

    print(f"[ml] Dataset size: {len(df_obs)} obs "
          f"({len(df_train)} train, {len(df_test)} test)")

    results: Dict[str, Dict[str, float]] = {}

    # 1) Plain linear regression baseline
    lin_model = LinearRegression(fit_intercept=True)
    results["LinearRegression"] = _evaluate_model(
        "LinearRegression",
        lin_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # 2) LassoCV (L1-regularised linear model)
    lasso = LassoCV(
        alphas=None,      # let the model choose a reasonable grid
        cv=5,
        random_state=cfg.random_state,
        max_iter=5000,
    )
    results["LassoCV"] = _evaluate_model(
        "LassoCV",
        lasso,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # 3) Random Forest (non-linear benchmark, controlled complexity)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=20,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    results["RandomForest"] = _evaluate_model(
        "RandomForest",
        rf,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # 4) XGBoost (if available)
    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        results["XGBRegressor"] = _evaluate_model(
            "XGBRegressor",
            xgb,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    else:
        print("\n[ml] XGBRegressor not available (xgboost not installed).")

    # 5) LightGBM (if available)
    if LGBMRegressor is not None:
        lgbm = LGBMRegressor(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.random_state,
        )
        results["LGBMRegressor"] = _evaluate_model(
            "LGBMRegressor",
            lgbm,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    else:
        print("\n[ml] LGBMRegressor not available (lightgbm not installed).")

    return results


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small ML benchmark on the same tiny US universe used in the rest of
    the project.

    We use a 5-day horizon (roughly weekly returns) to reduce some of the
    day-to-day noise. Absolute R^2 and IC values will still be small,
    which is expected in this kind of setup. The focus is on having a
    clean, well-structured pipeline rather than on overfitting the data.
    """
    universe = ["AAPL", "MSFT", "SPY", "META", "GOOGL"]

    cfg = MLModelConfig(
        tickers=universe,
        start="2015-01-01",
        end="2024-01-01",
        horizon=5,
        test_fraction=0.2,
        random_state=42,
    )

    results = run_ml_benchmark(cfg)

    print("\n[ml] Summary of test R^2 and IC:")
    for name, res in results.items():
        print(
            f"  {name:15s} -> Test R^2 = {res['test_r2']:.4e}, "
            f"Test IC = {res['test_ic']:.4e}"
        )
