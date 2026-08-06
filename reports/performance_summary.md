# Performance Metrics & Strategy Analytics Report

This research report presents a comparative performance evaluation of the **Momentum Crossover Strategy** and the **Rolling Z-Score Mean Reversion Strategy** backtested on the Nifty 50 index (2013-01-02 to 2024-12-31).

---

## 1. Executive Summary

We evaluated both strategies under a realistic next-day Open execution model (`next_open`) with 5 basis points of transaction fees and 2 basis points of slippage.

* **Momentum Crossover** outperforms on an absolute and risk-adjusted basis, achieving a **CAGR of 7.28%** and a **Sharpe Ratio of 0.6782**, with a maximum drawdown of **22.48%**.
* **Mean Reversion Z-Score** achieves a very high **Win Rate of 79.59%**, but suffers from severe tail risk, resulting in a low **CAGR of 2.09%**, a **Sharpe Ratio of 0.2633**, and a deep maximum drawdown of **34.78%** from which the strategy has not yet recovered.

---

## 2. Quantitative Performance Scorecard

The table below provides a consolidated summary of key strategy metrics.

| Metric Group | Performance Metric | Momentum Crossover | Mean Reversion Z-Score |
|---|---|---|---|
| **Return Metrics** | Total Return (%) | 127.46% | 27.33% |
| | CAGR (%) | 7.28% | 2.09% |
| | Average Annual Return (%) | 7.34% | 2.14% |
| **Risk Metrics** | Annualized Volatility (%) | 11.31% | 9.65% |
| | Semi Deviation (%) | 0.50% | 0.44% |
| | Maximum Drawdown (%) | 22.48% | 34.78% |
| | Ulcer Index (%) | 7.74% | 16.73% |
| **Risk-Adjusted Ratios** | Sharpe Ratio | 0.6782 | 0.2633 |
| | Sortino Ratio | 0.9530 | 0.3500 |
| | Calmar Ratio | 0.3239 | 0.0600 |
| **Trade Statistics** | Total Trades | 16 | 49 |
| | Win Rate (%) | 56.25% | 79.59% |
| | Profit Factor | 3.5115 | 1.7431 |
| | Expectancy per Trade (%) | 1.94% | 0.22% |
| **Benchmark Metrics** | Beta vs. Nifty 50 | 0.4684 | 0.3440 |
| | Alpha (Annualized %) | 1.53% | -1.97% |
| | Up Capture Ratio (%) | 61.06% | 19.10% |
| | Down Capture Ratio (%) | 61.46% | 19.05% |
| | Correlation with Benchmark | 0.6120 | 0.5262 |

---

## 3. Drawdown Event Analysis

### Momentum Strategy Top Worst Drawdowns
1. **Drawdown Depth: 22.48%** | Start: 2021-10-19 | Peak: 2021-10-18 | Recovery: 2024-06-03 | Duration: 661 trading days.
   * *Regime*: Post-COVID Nifty growth exhaustion and rate-hike correction.
2. **Drawdown Depth: 16.23%** | Start: 2015-03-04 | Peak: 2015-03-03 | Recovery: 2016-09-02 | Duration: 370 trading days.
   * *Regime*: Emerging markets growth slowdown.
3. **Drawdown Depth: 15.73%** | Start: 2013-05-29 | Peak: 2013-05-28 | Recovery: 2014-05-19 | Duration: 247 trading days.

### Mean Reversion Strategy Top Worst Drawdowns
1. **Drawdown Depth: 34.78%** | Start: 2020-02-27 | Peak: 2020-02-26 | Recovery: Active (Unrecovered) | Duration: 1,196+ trading days.
   * *Regime*: Triggered by the COVID-19 crash. The strategy bought index components during the rapid sell-off as Z-scores crossed below $-2.0$. Because the strategy lacked a stop-loss or trend filter, it remained locked in a long position as the market collapsed $38\%$.
2. **Drawdown Depth: 10.68%** | Start: 2018-09-17 | Peak: 2018-09-14 | Recovery: 2020-01-09 | Duration: 335 trading days.
3. **Drawdown Depth: 7.19%** | Start: 2016-11-09 | Peak: 2016-11-08 | Recovery: 2017-09-21 | Duration: 222 trading days.

---

## 4. Key Qualitative Observations

### 1. The Win Rate Paradox in Mean Reversion
The Mean Reversion strategy displays an impressive **Win Rate of 79.59%** (4 out of 5 trades are profitable). However, its cumulative return is poor (27.33% total return vs. 127.46% for Momentum). This is a classic hallmark of mean-reverting profiles: the strategy wins small amounts frequently but suffers from a "negative skew" tail event (the COVID crash drawdown of 34.78%) that erases years of small profits.

### 2. Market Exposure and Friction Drag
* **Momentum**: Exposure is $71.61\%$, with only 16 trades over 11 years. Frictional costs (₹3,113.64) represent a minor drag.
* **Mean Reversion**: Exposure is $12.10\%$, meaning the portfolio is in cash $87.90\%$ of the time. However, due to higher trade count (49 trades), it paid ₹7,845.07 in frictional costs, representing a significant return drag.

### 3. Alpha and Benchmark Capture
* **Momentum** captures $61.06\%$ of Nifty's upside while matching $61.46\%$ of Nifty's downside. Its Beta is $0.47$, yielding positive annualized alpha ($1.53\%$).
* **Mean Reversion** fails to generate alpha ($-1.97\%$). It captures only $19.10\%$ of upside while capturing $19.05\%$ of downside, indicating it does not effectively exploit the Nifty index's growth tailwind.

---

## 5. Strategic Strengths & Weaknesses

### Momentum Crossover
* **Strengths**: Captures major index growth trends, low trade churn (low frictional costs), high Profit Factor ($3.51$).
* **Weaknesses**: High market exposure ($71.6\%$) makes it vulnerable to overnight gaps and index corrections; moderate win rate ($56.25\%$) requires discipline during trendless periods.

### Mean Reversion Z-Score
* **Strengths**: High win rate ($79.59\%$), low time-in-market exposure ($12.10\%$), provides diversification as its returns are less correlated with standard momentum.
* **Weaknesses**: Unhedged downside risk (vulnerable to sharp regime sell-offs), high turnover increases trading fee drag, negative alpha.

---

## 6. Recommendations for Platform Improvements
1. **Downside Risk Protection (Stop-Loss)**: The Mean Reversion strategy requires a hard stop-loss (e.g. exit if Z-score falls below $-3.5$) to prevent catastrophic drawdown events like the 2020 COVID crash.
2. **Trend Filter**: Implement a macro trend filter (e.g., only enter Mean Reversion long trades when the Nifty index is above its 200-day moving average) to avoid buying during structural bear markets.
3. **Volatility-Adjusted Band Sizing**: Dynamically scale the entry Z-score threshold based on market volatility (e.g., widen thresholds during high-VIX regimes).
