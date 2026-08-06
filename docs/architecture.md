# System Architecture Documentation

This document describes the technical architecture, data pipelines, module dependencies, and execution flow of the **Systematic Momentum & Mean-Reversion Strategy Backtester**.

---

## 1. High-Level Pipeline Workflow

The platform follows a modular, state-driven workflow where data is ingested, cleaned, processed into signals, simulated via a vectorized execution engine, and analyzed for performance, validation, and market regimes.

```mermaid
graph TD
    A[Step 1: Data Acquisition] --> B[Step 2: Data Preprocessing]
    B --> C[Step 3: Exploratory Data Analysis]
    C --> D[Step 4: Signal Construction]
    D --> E[Step 5: Vectorized Backtest Engine]
    E --> F[Step 6: Performance & Analytics]
    F --> G[Step 7: IS/OOS & Robustness Validation]
    G --> H[Step 8: Market Regime Analysis]
```

---

## 2. Module Dependency Graph

The platform's source code under `src/` is decoupled into isolated layers:

```mermaid
graph TD
    main.py[main.py] --> src_regime[src/regime.py]
    main.py --> src_comparison[src/comparison.py]
    main.py --> src_robustness[src/robustness.py]
    main.py --> src_validation[src/validation.py]
    main.py --> src_analytics[src/analytics.py]
    
    src_regime --> src_regime_det[src/regime_detector.py]
    src_regime --> src_regime_met[src/regime_metrics.py]
    src_regime --> src_regime_vis[src/regime_visualization.py]
    
    src_comparison --> src_risk[src/risk.py]
    src_comparison --> src_metrics[src/metrics.py]
    src_comparison --> src_robustness
    
    src_validation --> src_engine[src/engine.py]
    src_validation --> src_momentum[src/momentum.py]
    src_validation --> src_mr[src/mean_reversion.py]
    
    src_robustness --> src_validation
    
    src_engine --> src_portfolio[src/portfolio.py]
    src_engine --> src_execution[src/execution.py]
    src_engine --> src_costs[src/costs.py]
```

---

## 3. Data Flow Architecture

The data flows from raw market values into structured risk adjusted profiles:

```
[Raw YFinance CSV] 
       ↓ (validate schema, handle gaps)
[Cleaned Parquet]
       ↓ (apply look-ahead bias shift)
[Execution Signals]
       ↓ (apply costs/slippage metrics)
[Daily Portfolio & Trade Book Log]
       ↓ (IS/OOS Splits & Regime Labels)
[Validation Scorecards & 60 Figures]
```

---

## 4. Execution Sequence Timeline

When `main.py` is invoked, the execution flow runs sequentially as follows:

```mermaid
sequenceDiagram
    participant CLI as main.py
    participant Pre as Preprocessor
    participant Sig as SignalGenerator
    participant Eng as BacktestEngine
    participant Val as ValidationEngine
    participant Reg as RegimeAnalyzer

    CLI->>Pre: Load raw price data & preprocess
    Pre-->>CLI: Return clean DataFrame
    CLI->>Sig: Construct Momentum & Mean Reversion signals
    Sig-->>CLI: Return signals DataFrame
    CLI->>Eng: Backtest full-history baseline strategies
    Eng-->>CLI: Return portfolio value history & trade books
    CLI->>Val: Run chronological splits (2013-19 vs. 2020-24) & Grid Search
    Val-->>CLI: Return frozen parameters & sensitivity scores
    CLI->>Reg: Run Market Regime Detector & Transition Matrix
    Reg-->>CLI: Return performance by state & transitions drift
    CLI->>CLI: Export final dashboards, CSV reports, and exit
```

---

## 5. Description of Core Modules

### 1. Data & Preprocessing Layer
* **`src/data_loader.py`**: Downloads ticker history from Yahoo Finance, validates columns, and handles data preservation.
* **`src/preprocessing.py`**: Cleans pricing series, checks for missing data, and exports standardized Parquet datasets.
* **`src/indicators.py`**: Computes rolling metrics (Averages, standard deviations, realized volatility, rolling drawdown series).

### 2. Signal Generation Layer
* **`src/momentum.py`**: Generates Dual Moving Average Crossover signals. Shifts signals by 1 day to prevent look-ahead bias.
* **`src/mean_reversion.py`**: Implements Z-Score bands with an entry/exit Finite State Machine (FSM) to handle position transitions.

### 3. Backtesting & Execution Layer
* **`src/engine.py`**: Unified interface for vectorized simulations.
* **`src/execution.py`**: Maps signals to physical shares using `next_open` or `close` execution assumptions.
* **`src/costs.py`**: Computes transaction friction (bps) and slippage on both entry and exit legs.
* **`src/portfolio.py`**: Calculates equity curves, daily returns, and compounding.

### 4. Performance & Validation Layer
* **`src/metrics.py` & `src/risk.py`**: Computes Sharpe, Sortino, Calmar, Max Drawdown duration, volatility, and benchmark beta.
* **`src/validation.py`**: Partitions data chronologically and manages parameter sweep grids.
* **`src/robustness.py`**: Executes sensitivity sweeps in parameter neighborhoods and computes Generalization and Stability scores.
* **`src/comparison.py`**: Formats side-by-side comparison tables and generates robustness plots.

### 5. Market Regime Layer
* **`src/regime_detector.py`**: Classifies trend, volatility, and crash/recovery states.
* **`src/regime_metrics.py`**: Computes performance on synthetic contiguous return series by regime.
* **`src/regime_visualization.py`**: Creates heatmaps, calendars, and comprehensive dashboards.
