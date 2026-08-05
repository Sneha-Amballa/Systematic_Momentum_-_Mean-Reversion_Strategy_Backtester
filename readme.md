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
├── src/
│   ├── data_loader.py  # Step 1: Data download, MultiIndex cleaning, raw validation
│   ├── preprocessing.py# Step 2: Data Preprocessor pipeline (11 stages)
│   ├── validators.py   # Step 2: Structural validation & data quality checks
│   └── utils.py        # System utilities (logging config, directories checking)
│
├── logs/
│
├── notebooks/          # Exploratory Data Analysis & signal design
├── requirements.txt    # Python dependencies
└── main.py             # Pipeline orchestrator
```

---

## Step 1: Data Acquisition

Downloads daily historical market data for the Nifty 50 Index (`^NSEI`) from Yahoo Finance, validates it, and saves it locally.
* **Period**: `2013-01-01` to `2024-12-31` (using exclusive end date `2025-01-01` to fetch final day).
* **Storage**: CSV (`data/raw/nifty50_raw.csv`) and Parquet (`data/raw/nifty50_raw.parquet`).

---

## Step 2: Data Cleaning & Preprocessing

Cleanses the raw historical market data, checks for quality anomalies, filters columns, optimizes memory footprint, and engineers basic features.

### Preprocessing Pipeline Stages
1. **Stage 1 — Load Raw Dataset**: Loads raw parquet data and logs baseline metrics (shape, date ranges).
2. **Stage 2 — Structural Validation**: Asserts DataFrame is not empty, index is a `DatetimeIndex`, sorted chronologically, unique (no duplicates), and required columns exist.
3. **Stage 3 — Data Quality Checks**: Scans and reports duplicate rows, duplicate timestamps, missing values, infinite values, negative/zero prices, and illogical high/low price overlaps. Logs all issues.
4. **Stage 4 — Handle Missing Data**: Drops rows with missing price values (`Open`/`High`/`Low`/`Close`). Explicitly **avoids forward-filling or calendar day reindexing** to prevent fabricating trading days and preserve natural market closures (weekends/NSE holidays).
5. **Stage 5 — Column Filter**: Configurable removal of the index volume column (`keep_volume=False` by default) because trading volumes are unreliable for market index instruments.
6. **Stage 6 — Data Type Optimization**: Casts price fields to standard `float64` for numerical precision and reports memory savings.
7. **Stage 7 — Feature Engineering**: Computes basic mathematical parameters:
   * **Daily Simple Return**: $Close_t / Close_{t-1} - 1$
   * **Daily Log Return**: $\ln(Close_t / Close_{t-1})$
   * **Daily Price Change**: $Close_t - Open_t$
   * **Daily Range**: $High_t - Low_t$
   * **Percentage Range**: $(High_t - Low_t) / Close_t$
8. **Stage 8 — Outlier Inspection**: Flags and reports days with anomalous returns exceeding a threshold (default $\pm 8\%$) without removing them, as financial tail events are critical for risk models.
9. **Stage 9 — Final Validation**: Ensures clean output preserves sorting, uniqueness, completeness (no NaNs), and schema dtypes.
10. **Stage 10 — Save Processed Dataset**: Saves clean data to CSV (`data/processed/nifty50_clean.csv`) and Parquet (`data/processed/nifty50_clean.parquet`).
11. **Stage 11 — Preprocessing Quality Report**: Emits a console summary logging row count changes, dropped items, created features, outlier count, and memory footprint difference.

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
Execute the main entry point to run data acquisition (if raw file is missing) and preprocessing pipeline:
```bash
python main.py
```

### 4. Outputs Generated
* **Logs**: `data_acquisition.log` (audit trail)
* **Raw Folder**: `data/raw/nifty50_raw.csv` and `nifty50_raw.parquet`
* **Processed Folder**: `data/processed/nifty50_clean.csv` and `nifty50_clean.parquet`