# Systematic Momentum & Mean-Reversion Strategy Backtester

A **from-scratch, vectorized backtesting framework** built in Python to research, test, and critically evaluate simple systematic trading signals on **Nifty 50 index data**.

The primary objective of this project is **not to find alpha**, but to build a rigorous quantitative research pipeline and understand why trading strategies often fail to generalize in live markets. The project focuses on key quantitative research concepts such as **overfitting, data-snooping bias, and regime dependence**.

---

## Project Overview

The project uses **10+ years of historical Nifty 50 market data** to design, implement, and evaluate two widely used systematic trading strategies:

- **Momentum Strategy** using a Moving Average Crossover
- **Mean-Reversion Strategy** using a Rolling Z-Score

Instead of relying on existing backtesting libraries, the project implements a **custom vectorized backtesting engine** using Pandas and NumPy. The engine models realistic trading assumptions such as **transaction costs**, **slippage**, and **signal lag**, providing a transparent view of the strategy evaluation process.

To assess strategy robustness, performance is evaluated using an **in-sample (training)** and **out-of-sample (testing)** split. Key performance metrics such as **Sharpe Ratio**, **Maximum Drawdown**, and overall returns are compared across both periods to understand whether the observed performance generalizes beyond the training data.

The project also examines **market regime dependence** by analyzing strategy performance during **trending** and **range-bound** market conditions, helping determine whether returns result from a genuine systematic edge or favorable historical conditions.

---

## Key Features

- Download and process **10+ years of historical Nifty 50 data**
- Build Momentum and Mean-Reversion trading signals
- Develop a **vectorized backtesting engine from scratch**
- Model realistic trading assumptions:
  - Transaction Costs
  - Slippage
  - Signal Lag
- Perform **In-Sample vs Out-of-Sample** evaluation
- Calculate quantitative performance metrics
- Analyze strategy performance across different market regimes
- Visualize equity curves, drawdowns, and strategy performance

---

## Trading Strategies

### Momentum Strategy
- Moving Average Crossover
- Generates long positions when the short-term moving average crosses above the long-term moving average.

### Mean-Reversion Strategy
- Rolling Z-Score based strategy
- Assumes that prices deviating significantly from their historical mean tend to revert back toward the average.

---

## Performance Evaluation

The backtesting framework evaluates strategies using:

- Sharpe Ratio
- Maximum Drawdown
- Cumulative Returns
- Win Rate

Strategies are evaluated using:

- **In-Sample (Training)**
- **Out-of-Sample (Testing)**

to assess model robustness and reduce the risk of overfitting.

---

## Research Focus

This project emphasizes critical evaluation of systematic trading strategies through:

- Overfitting Analysis
- Data-Snooping Bias
- Regime Dependence
- Transaction Cost Modeling
- Slippage Analysis
- Out-of-Sample Validation

---

## Tech Stack

- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- yfinance
- Jupyter Notebook

---

## License

This project is intended for educational and research purposes.
