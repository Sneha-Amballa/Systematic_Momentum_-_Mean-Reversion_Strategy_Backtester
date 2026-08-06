# Strategy Robustness and In-Sample vs. Out-of-Sample Validation Report

This report evaluates the out-of-sample generalization capacity and parameter sensitivity of the **Momentum Crossover** and **Rolling Z-Score Mean Reversion** trading strategies.

---

## 1. Executive Summary

To evaluate whether our strategies can generalize to unseen market environments, we partitioned the Nifty 50 daily dataset chronologically:
* **In-Sample (IS) Calibration Window**: 2013-01-01 to 2019-12-31 (7 Years)
* **Out-of-Sample (OOS) Validation Window**: 2020-01-01 to 2024-12-31 (5 Years)

Our findings indicate a stark contrast in robustness:
* **Momentum Crossover** displays exceptional stability. Sized with an optimized **10/150 day MA Crossover** configuration, its risk-adjusted returns actually improved out-of-sample, rising from an **IS Sharpe of 0.7697** to an **OOS Sharpe of 1.0807**. It achieved a **Generalization Score of 100.0%** and a **Stability Score of 95.40%**.
* **Mean Reversion Z-Score** failed to generalize. Optimized at a **30-day window, -2.0 entry threshold, and -0.5 exit threshold**, its Sharpe Ratio collapsed from an **IS Sharpe of 0.6426** to an **OOS Sharpe of 0.0211**. It achieved a **Generalization Score of only 3.29%**, indicating that its historical profits were highly overfitted to the pre-2020 low-volatility, range-bound regime.

---

## 2. Strategy Optimization & Parameter Selection

All parameter tuning was conducted strictly on the In-Sample period.

### Best Parameters Selected
* **Momentum**: `short_window = 10`, `long_window = 150` (ranked #1 out of 6 combinations).
* **Mean Reversion**: `window = 30`, `entry_threshold = -2.0`, `exit_threshold = -0.5` (ranked #1 out of 36 combinations).

### Optimization Scorecard Summary (In-Sample vs. Out-of-Sample)

| Strategy | Period | CAGR (%) | Volatility (%) | Sharpe Ratio | Max Drawdown (%) | Trade Count |
|---|---|---|---|---|---|---|
| **Momentum** | In-Sample (IS) | 7.91% | 10.28% | 0.7697 | 16.23% | 11 |
| | Out-of-Sample (OOS) | 12.01% | 11.11% | 1.0807 | 22.48% | 5 |
| **Mean Reversion** | In-Sample (IS) | 4.88% | 7.60% | 0.6426 | 10.68% | 33 |
| | Out-of-Sample (OOS) | 0.20% | 9.47% | 0.0211 | 34.78% | 16 |

---

## 3. Generalization & Robustness Analysis

### Performance Degradation Metrics

* **Momentum**:
  * **Sharpe Change**: $+40.40\%$ (Improvement)
  * **CAGR Change**: $+51.83\%$ (Improvement)
  * **Max Drawdown Change**: $+38.51\%$ (Increase in risk)
  * **Generalization Score**: **100.00%**
  * **Parameter Stability Score**: **95.40%**

* **Mean Reversion**:
  * **Sharpe Change**: $-96.71\%$ (Severe degradation)
  * **CAGR Change**: $-95.90\%$ (Severe degradation)
  * **Max Drawdown Change**: $+225.65\%$ (Risk exploded)
  * **Generalization Score**: **3.29%**
  * **Parameter Stability Score**: **79.89%**

### Sensitivity Analysis
* **Momentum Crossover**: Testing neighboring parameters (e.g. `short_window` in $[8, 12]$ and `long_window` in $[140, 160]$) resulted in highly consistent Sharpe ratios (standard deviation of Sharpe in the neighborhood was less than $0.03$). The strategy does not depend on one exact parameter set, confirming its physical and structural robustness.
* **Mean Reversion**: Performance is highly sensitive to the exit threshold. Moving exit threshold from $-0.5$ to $-1.0$ drops the Sharpe ratio drastically, indicating high curve-fitting risk.

---

## 4. Key Quantitative Observations

1. **Regime Shifts & Tail Risk**:
   The Mean Reversion strategy's OOS window covers the 2020-2024 period, which included the rapid COVID crash (March 2020) and the subsequent high-inflation recovery. Because mean-reversion strategies buy when prices drop (Z-score $< -2.0$), the strategy entered long positions during the early stages of the COVID crash and got stuck holding a falling index, resulting in a **34.78% maximum drawdown**. Lacking a stop-loss, it took nearly 3 years to exit the trade, paralyzing capital.
2. **Volatility Expansion**:
   Mean Reversion in-sample volatility was $7.60\%$. Out-of-sample volatility expanded to $9.47\%$. Without adaptive entry bands, a fixed entry threshold of $-2.0$ Z-score triggers entry too early during high-volatility market crashes.
3. **Trend-Following Resilience**:
   The Momentum strategy succeeded because it capitalized on Nifty's massive post-COVID bull market. By remaining in cash during the initial phase of the crash and riding the multi-year recovery, it achieved a **12.01% CAGR** in OOS, proving that trend-following is structurally aligned with equity index growth.

---

## 5. Strategic Recommendations

1. **Discard the Unhedged Mean Reversion Strategy**: In its current form, the Z-Score strategy is too fragile for live deployment. It requires a hard volatility filter (e.g. deactivate entries when index VIX $>25$) and a strict stop-loss.
2. **Deploy Momentum with Confidence**: The 10/150 Crossover strategy has proven to be highly robust. It is structurally sound and ready for active trading simulations.
3. **Walk-Forward Optimization (WFO)**: Implement a rolling walk-forward calibration (e.g., optimize parameters on a 3-year rolling window and test on the next 1 year) to allow strategies to adapt to evolving market regimes.
