# Methodology Documentation

This document explains the mathematical foundations, trading strategies, simulation models, validation protocols, and regime detection rules of the **Systematic Momentum & Mean-Reversion Strategy Backtester**.

---

## 1. Research Objective & Dataset

The objective is to construct, backtest, and validate two systematic trading models on index pricing data.
* **Dataset**: Nifty 50 historical daily close prices (2013-01-01 to 2024-12-31).
* **Data Preprocessing**: Checks schema validity, validates that timestamps are monotonic and sequential, handles gaps using forward-fill, and saves clean Parquet files.

---

## 2. Trading Strategy Formulation

### A. Momentum Moving Average Crossover
Momentum models capture trend persistence. We implement a Dual Simple Moving Average (SMA) crossover system:

1. **Short SMA ($SMA_S(t)$)**:
   $$SMA_S(t) = \frac{1}{S} \sum_{i=0}^{S-1} P(t-i)$$
2. **Long SMA ($SMA_L(t)$)**:
   $$SMA_L(t) = \frac{1}{L} \sum_{i=0}^{L-1} P(t-i)$$
3. **Signal Generation**:
   $$\text{Raw Signal}(t) = \begin{cases} 1 & \text{if } SMA_S(t) > SMA_L(t) \\ 0 & \text{otherwise} \end{cases}$$
4. **Look-Ahead Prevention**:
   $$\text{Execution Signal}(t) = \text{Raw Signal}(t-1)$$
   *By shifting the raw signal by 1 day, we ensure that a signal computed at the close of day $t-1$ is executed at the start (or close) of day $t$.*

### B. Mean Reversion Z-Score
Mean reversion models capture price deviations from a rolling average.

1. **Rolling Mean ($\mu(t)$)** and **Standard Deviation ($\sigma(t)$)**:
   $$\mu(t) = \frac{1}{W} \sum_{i=0}^{W-1} P(t-i)$$
   $$\sigma(t) = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} (P(t-i) - \mu(t))^2}$$
2. **Rolling Z-Score ($Z(t)$)**:
   $$Z(t) = \frac{P(t) - \mu(t)}{\sigma(t)}$$
3. **Signal Finite State Machine (FSM)**:
   * **State = 0 (No position)**: Enter long position if $Z(t) \le \text{Entry Threshold}$ (e.g. $-2.0$). Transition State to 1.
   * **State = 1 (Long position)**: Exit long position if $Z(t) \ge \text{Exit Threshold}$ (e.g. $-0.5$). Transition State to 0.
4. **Look-Ahead Prevention**:
   $$\text{Execution Signal}(t) = \text{Raw Signal}(t-1)$$

---

## 3. Backtesting & Execution Engine

We implement a vectorized backtesting engine with realistic operational parameters:

1. **Execution Model**:
   * `next_open`: Executes trades at the Open price of day $t$ using the signal from day $t-1$.
   * `close` (Default): Executes trades at the Close price of day $t$.
2. **Friction Model**:
   * **Transaction Costs**: Fixed rate (e.g., 5.0 bps) applied to the traded value.
   * **Slippage**: Flat fee (e.g., 2.0 bps) added to transaction costs on entry and exit legs.
   $$\text{Total Frictional Costs}(t) = (\text{Cost Bps} + \text{Slippage Bps}) \times |Position(t) - Position(t-1)| \times Price(t)$$

---

## 4. In-Sample vs. Out-of-Sample Validation

To validate strategy robustness and prevent curve fitting, we partition the dataset chronologically:
* **In-Sample (IS)**: 2013-01-01 to 2019-12-31. Used for grid-search parameter optimization.
* **Out-of-Sample (OOS)**: 2020-01-01 to 2024-12-31. Locked for final evaluation.

### A. Generalization Score
Measures Sharpe ratio preservation:
$$\text{Generalization Score} = \begin{cases} 100.0 & \text{if } Sharpe_{OOS} \ge Sharpe_{IS} \\ 0.0 & \text{if } Sharpe_{OOS} \le 0.0 \\ \frac{Sharpe_{OOS}}{Sharpe_{IS}} \times 100.0 & \text{otherwise} \end{cases}$$

### B. Parameter Stability Score
Calculates stability within a parameter neighborhood $N(P)$ on In-Sample data:
$$\text{Stability Score} = \max\left(0.0, \left(1.0 - \min\left(1.0, \frac{\sigma_{Sharpe}}{\mu_{Sharpe}}\right)\right) \times 100.0\right)$$

---

## 5. Market Regime Detection Rules

We classify daily index environments using trend and volatility filters:
* **Trend State**:
  * **Trending Up**: Close > 200d SMA AND 50d SMA > 200d SMA.
  * **Trending Down**: Close < 200d SMA AND 50d SMA < 200d SMA.
  * **Sideways**: Otherwise.
* **Volatility State**: Realized 20-day annualized volatility mapped to quantiles (33% and 66%) calculated using an expanding window.
* **Crash Overlay**: Drawdown from index peak > 15% AND rolling 20-day return < -10%.
* **Recovery Overlay**: Drawdown from index peak > 10% AND rolling 20-day return > 0.0%.
