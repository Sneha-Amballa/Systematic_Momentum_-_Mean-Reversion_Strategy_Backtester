# Systematic Momentum & Mean-Reversion Strategy Backtester

A professional quantitative research backtester designed to evaluate momentum and mean-reversion trading strategies on historical daily data for the Nifty 50 Index (`^NSEI`). The project is structured modularly to ensure clean separation of concerns and reproducibility.

---

## Project Structure

```text
project/
│
├── data/
│   ├── raw/            # Raw datasets downloaded from source (CSV, Parquet)
│   │     nifty50_raw.parquet
│   │
│   └── processed/      # Preprocessed, cleaned, and feature-engineered datasets
│         nifty50_clean.parquet
│         nifty50_clean.csv
│
├── notebooks/          # Exploratory Data Analysis & signal design
│     01_exploratory_data_analysis.ipynb
│
├── reports/            # Research summaries and generated figures
│   ├── eda_summary.md  # Compiled quants findings report
│   └── figures/        # High-DPI publication-quality plots (Matplotlib)
│         01_price_time_series.png
│         06_returns_distribution_plots.png
│         07_rolling_volatility_comparison.png
│         12_monthly_returns_heatmap.png
│         15_market_regime_exploration.png
│         ...
│
├── src/
│   ├── data_loader.py  # Step 1: Data download, MultiIndex cleaning, raw validation
│   ├── preprocessing.py# Step 2: Data Preprocessor pipeline (11 stages)
│   ├── validators.py   # Step 2: Structural validation & data quality checks
│   ├── eda.py          # Step 3: Statistical calculations and visualizations
│   └── utils.py        # System utilities (logging config, directories checking)
│
├── logs/
│
├── requirements.txt    # Python dependencies
└── main.py             # Pipeline orchestrator
```

---

## Step 1: Data Acquisition

Downloads daily historical market data for the Nifty 50 Index (`^NSEI`) from Yahoo Finance, validates it, and saves it locally.
* **Period**: `2013-01-01` to `2024-12-31`.
* **Storage**: CSV (`data/raw/nifty50_raw.csv`) and Parquet (`data/raw/nifty50_raw.parquet`).

---

## Step 2: Data Cleaning & Preprocessing

Cleanses the raw historical market data, checks for quality anomalies, filters columns, optimizes memory footprint, and engineers basic features.
* **Pruned Data**: Duplicates removed, columns trimmed (`Volume` dropped by default), type optimized to `float64`.
* **Trading Integrity**: No calendar day fabrication or forward-filling of prices.
* **Basic Features Engineered**: Simple returns, log returns, close-open price change, absolute high-low range, and percentage high-low range.

---

## Step 3: Exploratory Data Analysis (EDA)

Evaluates dataset structures, calculates descriptive statistics, runs statistical stationarity and normality tests, analyzes calendar seasonalities, and plots price/volatility behaviors.

### Analytical Operations
1. **Section 1 & 2: Dataset Overview & Quality**: Reports shape, dtypes, memory footprint, null counts, duplicate checks, and completeness ratios.
2. **Section 3: Descriptive Statistics**: Calculates mean, median, standard deviation, interquartile range (IQR), and coefficient of variation (CV) for all columns.
3. **Section 4: Price Behaviour Analysis**: Visualizes levels, intraday price range, daily changes, and cumulative compounded growth.
4. **Section 5: Returns Profile**: Analyzes return distribution via histograms, KDE fits, boxplots, violins, and Normal Q-Q plots.
5. **Section 6 & 10: Rolling Analysis**: Compares rolling 20, 50, and 100-day annualized volatilities and min-max boundaries.
6. **Section 7: Distribution Shape & Normality**:
   * Computes **Skewness** and **Excess Kurtosis** (leptokurtic confirmation).
   * Runs the **Jarque-Bera Test** to verify if return distributions follow a normal curve.
7. **Section 8: Stationarity Testing**:
   * Runs **Augmented Dickey-Fuller (ADF)** and **KPSS** tests on Close levels ($I(1)$ series) and Daily Returns ($I(0)$ series) to confirm stationarity requirements for time series modeling.
8. **Section 9: Correlation Matrices**: Generates correlation matrices, pair plots, and heatmaps.
9. **Section 11: Extreme Days**: Ranks top 20 positive/negative return days, intraday swings, and gap openings.
10. **Section 12, 13 & 14: Calendar Seasonality**:
    * Computes annual metrics (Return, Volatility, Max Drawdown, trading counts).
    * Compiles monthly compounded return heatmaps and monthly return boxplots.
    * Evaluates day-of-week returns distributions.
11. **Section 15: Market Regimes**: Identifies consolidation levels and shades historical event regimes (COVID-19 panic of 2020, 2016 consolidation, 2018 correction, 2022 global decline).

### Generated Report
All numerical outputs and quants findings are generated in the compiled research report **[reports/eda_summary.md](file:///c:/Users/USER/OneDrive/Desktop/Projects/Systematic%20Momentum%20&%20Mean-Reversion%20Strategy%20Backtester/reports/eda_summary.md)**.

---

## Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
Install packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline
Execute the main entry point to run data acquisition, preprocessing, and the full EDA suite:
```bash
python main.py
```

### 4. Notebook Exploration
Open the notebook **[notebooks/01_exploratory_data_analysis.ipynb](file:///c:/Users/USER/OneDrive/Desktop/Projects/Systematic%20Momentum%20&%20Mean-Reversion%20Strategy%20Backtester/notebooks/01_exploratory_data_analysis.ipynb)** in VS Code to review the statistical calculations and plots inline.