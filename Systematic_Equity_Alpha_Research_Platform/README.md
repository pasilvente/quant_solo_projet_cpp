# Systematic Equity Alpha Research Platform

This repository is a small **systematic equity research sandbox** I built to practise the full workflow a quant researcher would follow:

> raw market data → factor construction → alpha modelling (linear + ML) → portfolio construction → backtesting & diagnostics.

The goal is **not** to claim a magic trading strategy, but to have a clean, realistic framework that I can extend over time and use as a basis for further research.

---

## 1. High-level overview

The platform currently focuses on a small US equity universe (AAPL, MSFT, SPY, META, GOOGL) and implements:

- **Data layer**
  - Historical daily prices and volumes via `yfinance`
  - Disk caching in Parquet (`data/raw/` and `data/clean/`)

- **Factor layer**
  - Cross-sectional equity factors:
    - 12-month momentum excluding the last month (12M–1M)
    - 1-month momentum
    - 20d / 60d realised volatility
    - 20d average dollar volume (simple liquidity proxy)
  - Per-date **z-scores** of each factor (cross-sectional normalisation)

- **Alpha models**
  - A pooled **linear factor model**:
    r_{i,t+1} ≈ β_0 + β^T f_{i,t}
  - A **ML benchmark** on cross-sectional returns:
    - LinearRegression (baseline)
    - LassoCV (L1-regularised)
    - RandomForestRegressor
    - XGBRegressor (XGBoost)
    - LGBMRegressor (LightGBM)
  - Proper **time-based train/test split**, no shuffling across dates

- **Portfolio construction**
  - **Long/short equity book**, built from alpha scores:
    - long top X% of names by score
    - short bottom X%
    - weights scaled to be:
      - dollar-neutral (net exposure ≈ 0)
      - with a target gross exposure (e.g. 1.0)

- **Backtest engine**
  - Converts weights and realised returns into:
    - daily portfolio P&L
    - equity curve
    - performance statistics:
      - annualised return, volatility, Sharpe
      - max drawdown
      - average daily turnover

- **Analytics & tests**
  - Small helpers to format performance summaries
  - A few pytest-based checks on factor and backtest logic

Everything is written in **Python** using standard libraries:
numpy, pandas, yfinance, scikit-learn, xgboost, lightgbm, pytest.

---

## 2. Project structure

Inside the Systematic_Equity_Alpha_Research_Platform/ folder:

Systematic_Equity_Alpha_Research_Platform/
  data/
    raw/      # cached raw daily data (per ticker, Parquet)
    clean/    # factors and combined panels (Parquet)
  src/
    equity_alpha/
      __init__.py

      data/
        __init__.py
        data_loader.py   # download/cache prices & volumes
        factors.py       # factor construction

      models/
        __init__.py
        linear_factor_model.py   # simple pooled linear factor model
        ml_models.py             # ML models benchmark (Lasso, RF, XGB, LGBM)

      strategy/
        __init__.py
        portfolio_construction.py  # long/short weights from scores

      backtest/
        __init__.py
        engine.py       # end-to-end backtest engine
        analytics.py    # pretty printing of stats, weight diagnostics

      tests/
        __init__.py
        test_factors.py         # basic factor construction tests
        test_backtest_logic.py  # sanity checks on backtest engine

---

## 3. Data layer

### 3.1 data_loader.py

Responsibilities:

- Download daily OHLCV data from yfinance for a list of tickers
- Save each ticker as a Parquet file under data/raw/
- Build a panel of prices or volumes (Date × Ticker)
- Compute log-returns

Key ideas:

- I use a simple disk cache to avoid hitting the API repeatedly:
  - AAPL_daily.parquet, MSFT_daily.parquet, etc.
- All time series are aligned on a common calendar intersection.
- Returns are computed as log-returns, which add up more nicely in time.

Typical flow:

1. download_daily_prices(...)
   - Downloads data from yfinance (if not already cached)
   - Selects columns like ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
   - Saves them to data/raw/TICKER_daily.parquet

2. load_price_panel(tickers, field="Adj Close", ...)
   - Rebuilds a wide DataFrame with index = Date, columns = tickers.

3. compute_log_returns(prices, horizon=1)
   - Computes log(P_t / P_{t-1}) (or over horizon days if >1).

---

## 4. Factor layer

### 4.1 factors.py

This module constructs a set of classic cross-sectional equity factors for a given universe and date range, and stores them in data/clean/.

Implemented factors:

- Momentum
  - mom_12m_excl_1m : 12-month log-momentum excluding the last month (approx. 252 trading days lookback, with a 21-day lag)
  - mom_1m : 1-month momentum (21 trading days)

- Realised volatility
  - vol_20d : 20-day realised volatility (annualised)
  - vol_60d : 60-day realised volatility (annualised)

- Liquidity proxy
  - liq_20d_dv : 20-day average dollar volume (price × volume, then rolling mean)

For each factor, I also compute a cross-sectional z-score per date:

- z_mom_12m_excl_1m, z_mom_1m, z_vol_20d, z_vol_60d, z_liq_20d_dv

At each date t:
- subtract the cross-sectional mean across all tickers
- divide by the cross-sectional standard deviation (if enough names)

This gives factors on a comparable scale, which is helpful for both linear models and ML.

The function:

factors = build_factors(
    tickers=["AAPL", "MSFT", "SPY", "META", "GOOGL"],
    start="2015-01-01",
    end="2024-01-01",
    force_download=False,
)

will:

- ensure raw data is available,
- compute all factor panels,
- save each factor to data/clean/<name>.parquet,
- create a combined factors_panel.parquet with a MultiIndex on columns (factor_name, ticker).

---

## 5. Alpha models

### 5.1 Linear factor model (linear_factor_model.py)

This is a simple pooled cross-sectional regression:

r_{i,t+1} = β_0 + β^T f_{i,t} + ε_{i,t+1}

- Target: next-day log-return (or next-horizon return)
- Features: selected z-scored factors at time t:
  - z_mom_12m_excl_1m
  - z_mom_1m
  - z_vol_20d
  - z_vol_60d
  - z_liq_20d_dv

The model is global (same coefficients for all stocks and all dates in the sample). The goal is to get a cross-sectional ranking of expected returns, not to perfectly explain all variance.

Key function:

- LinearFactorModel.fit_from_raw(config):
  - automatically calls build_factors(...) if needed
  - builds a long-format dataset (Date, Ticker, factors, target)
  - fits pooled OLS via np.linalg.lstsq

- LinearFactorModel.predict_scores(tickers, start, end):
  - returns a score panel (Date × Ticker) of predicted returns,
  - used later for portfolio construction.

### 5.2 ML models (ml_models.py)

This module reuses the same factors, but benchmarks a few models from scikit-learn, XGBoost and LightGBM.

Workflow:

1. Build a long-format dataset:

   - columns: ["Date", "Ticker"] + factor_names + ["target"]
   - target = log-return over a horizon of 5 days (weekly approx.):
     this helps reduce a bit of day-to-day noise.

2. Split into train / test by time, not randomly:
   - early dates → train
   - most recent dates → test

3. Train and evaluate:

   - LinearRegression
   - LassoCV (L1-regularised linear model)
   - RandomForestRegressor
   - XGBRegressor (if xgboost is installed)
   - LGBMRegressor (if lightgbm is installed)

For each model I log:

- train / test R²
- train / test MSE
- test IC = correlation between predictions and realised returns

As expected for noisy weekly returns on a tiny universe with simple factors, test R² and IC are very small, sometimes slightly negative. That’s actually an interesting and honest takeaway: you see that with this setup, you cannot “beat” the noise easily, even with complex models.

---

## 6. Portfolio construction

### 6.1 strategy/portfolio_construction.py

Given a panel of alpha scores (predicted returns) with shape (Date × Ticker), I build a simple long/short market-neutral portfolio.

Logic at each date:

1. Drop tickers with missing scores
2. Sort remaining tickers by score (descending)
3. Select:
   - long bucket: top long_quantile fraction (e.g. 40%)
   - short bucket: bottom short_quantile fraction (e.g. 40%)
4. Assign raw equal weights:
   - longs: +1 / n_long each
   - shorts: −1 / n_short each
5. Rescale weights so that:
   - sum of absolute weights = gross_target (e.g. 1.0)
   - net exposure ≈ 0 (long and short legs are roughly balanced)

Returned object: a weight matrix with same index/columns as scores.

This kind of construction is simple but very common as a first pass in systematic equity research: it gives you a transparent mapping from “signal strength” to an implementable portfolio.

---

## 7. Backtest engine

### 7.1 backtest/engine.py

The backtest engine connects all the pieces:

1. Fit the linear factor model over the chosen period.
2. Compute scores for the full window.
3. Build long/short weights from the scores.
4. Load realised returns of the universe.
5. Compute portfolio returns using a realistic convention:
   - weights at date t are applied to returns from t+1,
   - i.e. decisions are taken at the close and executed on the next bar.

6. Build the equity curve: cumulative product of (1 + r_t)
7. Compute summary statistics:
   - annualised return
   - annualised volatility
   - Sharpe ratio
   - max drawdown
   - average daily turnover:
     turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|

The function run_backtest(BacktestConfig) returns a dictionary containing:

- scores   : DataFrame (Date × Ticker)
- weights  : DataFrame (Date × Ticker)
- port_rets: Series of daily portfolio returns
- equity   : Series of cumulative equity
- stats    : dict of performance metrics

Currently the universe is very small (5 names), so the performance numbers are not the focus; the important part is that the plumbing is correct and easy to extend.

---

## 8. Analytics & tests

### 8.1 backtest/analytics.py

Provides a couple of small helpers:

- format_stats(stats):
  - Formats the performance dictionary into a readable multi-line string.
- summarize_weights(weights):
  - Computes average net and gross exposure, and average number of non-zero positions per day.
  - Useful to sanity-check that the book behaves as expected.

### 8.2 tests/

I added a few pytest tests to check the basic logic:

- test_factors.py:
  - calls build_factors on a small universe,
  - asserts that factor panels are non-empty and have the expected columns.

- test_backtest_logic.py:
  - runs a short backtest on a small universe,
  - checks shapes of scores/weights/returns,
  - checks that net exposure is roughly neutral on average,
  - verifies that performance stats keys are present.

The idea is not to have a full industrial test suite, but to show that I’m used to writing minimal sanity checks around this kind of research code.

---

## 9. How to run

### 9.1 Environment

I use a local virtual environment:

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .\.venv\Scripts\Activate.ps1 # Windows PowerShell

pip install -r requirements.txt

(Dependencies: numpy, pandas, yfinance, pyarrow,
scikit-learn, xgboost, lightgbm, pytest.)

### 9.2 Examples

From inside Systematic_Equity_Alpha_Research_Platform/src:

- Build factors:

  python -m equity_alpha.data.factors

- Fit linear model + inspect scores:

  python -m equity_alpha.models.linear_factor_model

- Run ML benchmark (Linear, Lasso, RF, XGB, LGBM):

  python -m equity_alpha.models.ml_models

- Run end-to-end backtest:

  python -m equity_alpha.backtest.engine

- Run tests (from project root):

  pytest -q

---

## 10. Limitations and possible extensions

This is intentionally a compact, didactic project. Some obvious extensions:

- Larger universes
  - Move from 5 names to a proper large-cap universe (e.g. S&P 100 / 500 subset).

- Richer factor library
  - Add basic value, quality and low-risk factors.
  - Introduce sector / industry controls.

- More realistic risk & costs
  - Sector-neutral / beta-neutral portfolio construction.
  - Explicit transaction costs and slippage.
  - Position limits / turnover constraints.

- More advanced ML experiments
  - Compare different horizons (e.g. 5d vs 20d).
  - Use cross-validation schemes that respect time.

I see this repository as a base platform I can keep extending rather than a finished product. The key design choice was to keep the code modular and readable, so that adding new factors, models or portfolio rules is straightforward.
