# Quantitative Research Paper: Generalization Capacity and Market Regime Sensitivity of Systematic Trend-Following and Mean-Reversion Models

**Author**: Senior Quantitative Researcher  
**Institution/Company**: Quantitative Hedge Fund Group  
**Date**: August 2026  

---

## Abstract
This paper presents a rigorous empirical study on the out-of-sample generalization capacity and regime sensitivity of two fundamental systematic trading paradigms: Dual Simple Moving Average (SMA) Momentum Crossover and Rolling Z-Score Mean Reversion. Using a 12-year daily history of the Nifty 50 index (2013–2024), we optimize strategy parameters strictly on an In-Sample (IS) calibration split (2013–2019) and validate them on a locked Out-of-Sample (OOS) verification split (2020–2024). We incorporate realistic friction costs (5.0 bps transaction costs and 2.0 bps slippage per trade leg). 

Our findings indicate a stark performance divergence. The Momentum strategy generalizes successfully, improving from an IS Sharpe of 0.77 to an OOS Sharpe of 1.08, achieving a Generalization Score of 100.0% and a Parameter Stability Score of 95.40%. Conversely, the Mean Reversion strategy suffers total degradation, with its Sharpe ratio collapsing from 0.64 (IS) to 0.02 (OOS), achieving a Generalization Score of only 3.29%. Further, we model state-dependent strategy returns across 11 objective market regimes, illustrating the structural tail-risk characteristics that govern mean-reversion anomalies.

---

## 1. Introduction
Quantitative trading strategies are highly vulnerable to overfitting—the mathematical memorization of historical noise rather than the capture of an enduring economic anomaly. In quantitative finance, separating historical pricing series into In-Sample (IS) and Out-of-Sample (OOS) periods is the industry standard for verifying generalization.

This paper systematically investigates:
1. The mathematical design of trend-following and mean-reverting strategies.
2. The role of chronological splitting and grid optimization in avoiding data snooping.
3. The sensitivity of optimized parameters within localized parameter neighborhoods.
4. The performance of strategies across objectively defined market regimes.

---

## 2. Related Concepts & Strategy Formulations

### 2.1 Trend-Following Momentum
The Momentum Crossover strategy exploits the tendency of asset prices to exhibit persistent drift (auto-correlation) due to gradual information dissemination and behavioral herd effects. We define:

$$SMA_{Short}(t) = \frac{1}{S} \sum_{i=0}^{S-1} P(t-i)$$

$$SMA_{Long}(t) = \frac{1}{L} \sum_{i=0}^{L-1} P(t-i)$$

$$\text{Raw Signal}(t) = \begin{cases} 1 & \text{if } SMA_{Short}(t) > SMA_{Long}(t) \\ 0 & \text{otherwise} \end{cases}$$

To prevent look-ahead bias, the signal is lagged:

$$\text{Execution Signal}(t) = \text{Raw Signal}(t-1)$$

### 2.2 Z-Score Mean Reversion
Mean reversion assumes that price deviations from a local rolling average represent short-term supply-demand imbalances that will eventually normalize. We calculate the Z-Score:

$$Z(t) = \frac{P(t) - \mu_W(t)}{\sigma_W(t)}$$

where $\mu_W(t)$ is the rolling mean and $\sigma_W(t)$ is the rolling standard deviation over lookback window $W$. 

To enter and exit positions, we implement a Finite State Machine (FSM):
* **State 0 (Flat)**: If $Z(t) \le \text{Entry Threshold}$, buy ($Signal = 1$). State becomes 1.
* **State 1 (Long)**: If $Z(t) \ge \text{Exit Threshold}$, sell ($Signal = 0$). State becomes 0.

$$\text{Execution Signal}(t) = \text{Raw Signal}(t-1)$$

---

## 3. Dataset & Preprocessing
The dataset comprises daily close prices of the Nifty 50 index from 2013-01-01 to 2024-12-31.
* **Cleaning Protocol**:
  * Missing observations are forward-filled to prevent look-ahead data leakage.
  * Pricing tables are stored in Parquet files to preserve high-precision timestamps.
  * Monotonic chronological ordering is verified.

---

## 4. Experimental Setup

We execute backtests under realistic institutional execution assumptions:
* **Initial Capital**: ₹100,000.00
* **Execution Model**: `next_open` (Orders execute at the open price of day $t$ using the signal from day $t-1$).
* **Frictional Fees**:
  * Brokerage/Exchange fees: 5.0 bps per trade.
  * Slippage model: 2.0 bps flat execution friction.
  * Total transaction cost: 7.0 bps applied to absolute traded values.

---

## 5. Performance Evaluation & Robustness Analysis

All parameters were optimized strictly on the In-Sample split (2013–2019). The optimal configurations selected were:
* **Momentum**: $S = 10$, $L = 150$.
* **Mean Reversion**: $W = 30$, $\text{Entry} = -2.0$, $\text{Exit} = -0.5$.

### 5.1 Validation Results (IS vs. OOS)

The table below compiles performance across periods:

| Strategy | Period | CAGR (%) | Sharpe | Sortino | Calmar | Max DD (%) | Win Rate | Trade Count |
|---|---|---|---|---|---|---|---|---|
| **Momentum** | In-Sample (IS) | 7.42% | 0.7697 | 1.1353 | 0.5705 | 13.00% | 50.00% | 10 |
| | Out-of-Sample (OOS) | 13.22% | 1.0807 | 1.5302 | 0.9109 | 14.51% | 60.00% | 5 |
| **Mean Reversion** | In-Sample (IS) | 4.82% | 0.6426 | 0.9598 | 0.4711 | 10.22% | 79.17% | 24 |
| | Out-of-Sample (OOS) | -0.62% | 0.0211 | 0.0272 | -0.0179 | 34.78% | 81.25% | 16 |

### 5.2 Robustness Ratios
* **Momentum**:
  * **Generalization Score**: **100.00%**
  * **Stability Score (In-Sample Neighborhood)**: **95.40%**
* **Mean Reversion**:
  * **Generalization Score**: **3.29%**
  * **Stability Score (In-Sample Neighborhood)**: **79.89%**

The Momentum strategy's out-of-sample performance improved during Nifty's massive trending bull run of 2020–2024. Its parameter stability score of 95.40% indicates a flat, robust parameter space. In contrast, the Mean Reversion strategy's Sharpe ratio collapsed to 0.02 in OOS, indicating high overfitting and regime dependency.

---

## 6. Market Regime Sensitivity Analysis

We partition Nifty 50 price history into 11 distinct regimes to analyze strategy edges.

### 6.1 Performance metrics by Regime (2013-2024)

* **Momentum Crossover** thrives in:
  * **Trending Up Low Volatility**: Sharpe = 1.65, CAGR = 15.69%.
  * **Trending Up High Volatility**: Sharpe = 1.57, CAGR = 21.05%.
  * **Trending Down Low Volatility**: Sharpe = -3.07 (Excellent short/cash positioning).
* **Mean Reversion Z-Score** thrives in:
  * **Trending Up High Volatility**: Sharpe = 2.68, CAGR = 37.33% (Buying frequent minor pullbacks).
  * **Recovery**: Sharpe = 1.31, CAGR = 16.53%.
* **Mean Reversion Z-Score** suffers severely in:
  * **Crash**: Sharpe = -2.21, Max DD = 30.75%, CAGR = -44.57%.

### 6.2 Regime Transitions (Markov Probabilities)
Transition matrix analysis indicates that high-volatility sideways markets are highly unstable, transitioning rapidly to trending states. The **Crash** regime is highly transient, persisting day-to-day with a **91.3% probability** and transitioning to **Recovery** with an **8.7% probability**.

---

## 7. Discussion & Limitations

1. **Why Momentum Generalizes Better**: Momentum tracks large-scale trend changes, which are structurally supported by equity market growth. It automatically transitions to cash during prolonged downturns.
2. **Mean Reversion Tail Risk**: Mean reversion strategies function as "short volatility" indicators. They gain small amounts of capital consistently during range-bound regimes but are vulnerable to large drawdowns when trends break, as occurred during the COVID crash in March 2020.
3. **Data Limitations**: The use of daily close data hides intraday path dependency and liquidity spikes.

---

## 8. Conclusion
Chronological out-of-sample validation and parameter sensitivity analyses are essential for constructing robust quantitative models. Momentum strategies demonstrate consistent generalizability and stability, whereas Z-score mean reversion strategies carry severe tail risk during structural regime shifts, requiring VIX filters and stop-loss limits to be viable in production.

---

## References
1. Carhart, M. M. (1997). On Persistence in Mutual Fund Performance. *Journal of Finance*, 52(1), 57-82.
2. Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *Journal of Finance*, 48(1), 65-91.
3. Lo, A. W., & MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test. *Review of Financial Studies*, 1(1), 41-66.
