# Changelog

All notable changes to the **Systematic Momentum & Mean-Reversion Strategy Backtester** are documented in this file.

---

## [v1.0.0] - 2026-08-06
### Final Research Release & Publication Packaging
#### Features
* Standardized config setup including `requirements.txt`, `pyproject.toml`, `.gitignore`, and MIT `LICENSE`.
* Created comprehensive system diagrams mapping code layout, dependency structures, and execution trees.
* Published academic-style research reports detailing parameter robustness scorecards and market state performance matrices.
* Rewrote primary landing page (`README.md`) with badges, diagrams, and research highlights.

---

## [v0.9.0] - 2026-08-06
### Step 8: Market Regime Analysis
#### Features
* Implemented objective trend classification (Close > 200d SMA & 50d SMA > 200d SMA).
* Added look-ahead free volatility regime detection using expanding-window quantiles.
* Layered index event triggers to identify `Crash` (drawdown > 15%, returns < -10%) and `Recovery` (drawdown > 10%, returns > 0%) periods.
* Built Markov transition probability calculations and post-transition drift estimators.
* Generated 20 publication-quality regime visualizations, including a calendar and transition matrix heatmaps.

---

## [v0.8.0] - 2026-08-06
### Step 7: In-Sample vs. Out-of-Sample Validation
#### Features
* Implemented chronological splitting: In-Sample (2013–2019) and Out-of-Sample (2020–2024).
* Created parameter optimization sweep routines targeting Sharpe, Sortino, CAGR, and Calmar ratios.
* Built sensitivity checks evaluating parameter neighborhoods and calculating Stability and Generalization scores.
* Generated 20 robustness visualizations, including parameter surface heatmaps.

---

## [v0.7.0] - 2026-08-06
### Step 6: Performance Analytics Framework
#### Features
* Implemented metrics modules to compute CAGR, Sharpe, Sortino, Calmar, Volatility, and Max Drawdown.
* Created benchmarking modules to calculate Alpha, Beta, Information Ratio, and tracking errors.
* Developed trade logging to calculate win rate, profit factor, and expectancy.
* Built 20 analytics visualizations (drawdown timelines, returns distribution).

---

## [v0.6.0] - 2026-08-05
### Step 5: Vectorized Backtesting Engine
#### Features
* Implemented next-day open and same-day close execution models.
* Developed transaction cost and slippage estimation modules applied to absolute traded values.
* Created cash reconciliation logging to track portfolio values.

---

## [v0.5.0] - 2026-08-05
### Step 4B: Mean Reversion Strategy
#### Features
* Built lookback Z-Score indicator calculations.
* Implemented Finite State Machine (FSM) to manage position state transitions based on entry/exit thresholds.

---

## [v0.4.0] - 2026-08-05
### Step 4A: Momentum Strategy
#### Features
* Implemented Dual moving average crossover signal logic.
* Added 1-day execution signal lag to prevent look-ahead bias.

---

## [v0.3.0] - 2026-08-04
### Step 3: Exploratory Data Analysis
#### Features
* Conducted stationary tests (ADF).
* Plotted rolling volatility, autocorrelation (ACF/PACF), and daily returns distribution histograms.

---

## [v0.2.0] - 2026-08-04
### Step 2: Data Preprocessing
#### Features
* Created schema validations verifying pricing records.
* Implemented gap-filling routines using forward-fill.

---

## [v0.1.0] - 2026-08-04
### Step 1: Data Acquisition
#### Features
* Created downloader module utilizing `yfinance` to fetch Nifty 50 daily close history.
