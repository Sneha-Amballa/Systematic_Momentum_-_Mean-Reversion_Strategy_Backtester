# Systematic Mean-Reversion Strategy Research Summary

This report presents the theoretical foundation, signal construction methodology, empirical results, parameter sensitivity analysis, and qualitative assessment for the Rolling Z-Score Mean Reversion strategy.

---

## 1. Mean Reversion Theory

### What is Mean Reversion?
Mean reversion is a financial theory asserting that asset prices, returns, and valuation metrics tend to return to their historical long-term average (mean) after experiencing extreme deviations. 

### Why Financial Prices Revert
In liquid financial markets, price reversion is driven by several structural and behavioral factors:
1. **Market Microstructure and Liquidity Imbalances**: Large institutional buy/sell flows can temporarily exhaust local liquidity, pushing prices away from fundamental value. Once the order flow imbalance is resolved, prices revert.
2. **Behavioral Overreaction**: Investors frequently overreact to earnings releases, news events, or macroeconomic indicators, driving prices to irrational levels. Over time, rational arbitrageurs bring prices back to fair value.
3. **Index Rebalancing**: For equity indices like the Nifty 50, underperforming components are replaced with stronger ones, which structurally prevents the index from collapsing to zero, reinforcing a long-term upward mean-reverting behavior.

### Momentum vs. Mean Reversion
* **Momentum (Trend Following)**: Assumes that price trends persist over a medium-to-long term. It buys assets that are rising in price, expecting them to rise further. It performs best in persistent, high-volatility, macro-driven markets.
* **Mean Reversion**: Assumes that price deviations are transient noise. It buys assets that have fallen relative to their history (undervalued), expecting them to rebound. It performs best in choppy, range-bound, low-trend markets.

| Dimension | Momentum | Mean Reversion |
|---|---|---|
| **Underlying Hypothesis** | Trends persist ($P(\text{up} \mid \text{up}) > 0.5$) | Extremes revert ($P(\text{revert} \mid \text{extreme}) \to 1$) |
| **Market Regime** | Trending / High Dispersion | Range-Bound / Low Volatility |
| **Trading Style** | "Buy High, Sell Higher" | "Buy Low, Sell Average" |
| **Typical Win Rate** | Low ($30\% - 40\%$) | High ($60\% - 80\%$) |
| **Typical Payoff Profile** | Positive skew (small losses, huge wins) | Negative skew (small wins, potentially huge losses) |

### When Mean Reversion Works
* **Range-Bound Markets**: When there are no strong macroeconomic drivers, and prices swing between support and resistance.
* **Liquid Equity Indices**: Diversified indices naturally mean-revert due to component index weighting and regular survivorship bias rebalancing.
* **Cointegrated Pairs**: Pairs of economically linked assets (e.g., dual-class shares, commodity spreads) that share a long-term equilibrium.

### When Mean Reversion Fails
* **Strong Trends**: During strong macro-regimes (e.g., major bull or bear markets), prices can deviate by 3 or 4 standard deviations and remain there for months.
* **Individual Corporate Default / Bankruptcy**: An individual stock experiencing a structural shift (e.g., fraudulent earnings, disruption) will slide to zero, creating a "value trap" where standard deviation expansions do not lead to reversion.
* **Structural Regime Change**: A permanent change in the economic environment (e.g., interest rate hikes, regulatory changes) can establish a completely new mean level.

---

## 2. Indicator Parameters & Signal Logic

### Formulas
The strategy relies on a rolling window $W$ to calculate the following statistics for the close price series $P_t$:

1. **Rolling Mean** ($\mu_t$):
   $$\mu_t = \frac{1}{W} \sum_{i=0}^{W-1} P_{t-i}$$
2. **Rolling Standard Deviation** ($\sigma_t$):
   $$\sigma_t = \sqrt{\frac{1}{W-1} \sum_{i=0}^{W-1} (P_{t-i} - \mu_t)^2}$$
3. **Rolling Z-Score** ($Z_t$):
   $$Z_t = \frac{P_t - \mu_t}{\sigma_t}$$

### Signal Logic (Finite-State Machine)
To avoid churning positions, the strategy utilizes a stateful FSM:
* **Entry (Long)**: If we are Flat and the Z-Score crosses below the entry threshold ($Z_t < \text{Entry Threshold}$), we transition to Long.
* **Exit (Long)**: If we are Long, we remain invested until the Z-Score crosses above the exit threshold ($Z_t > \text{Exit Threshold}$), at which point we transition back to Flat.
* **Look-Ahead Bias Prevention**: The signal generated at the close of day $t$ (using $Z_t$) is executed on day $t+1$:
  $$\text{Execution\_Signal}_t = \text{Raw\_Signal}_{t-1}$$
  $$\text{Position}_t = \text{Execution\_Signal}_t$$

---

## 3. Empirical Results (Default Configuration)

The strategy was evaluated on the **Nifty 50 clean dataset** spanning from **2013-01-02 to 2024-12-31** (2,946 calendar rows; 2,926 active trading days after the 20-day warmup).

### Default Parameters
* **Rolling Window**: 20 days
* **Entry Threshold**: -2.0
* **Exit Threshold**: -0.5

### Strategy Trade Statistics
The following table summarizes the signal properties:

| Metric | Empirical Value |
|---|---|
| **Total Active Trading Days** | 2,926 |
| **Number of Trades** | 49 |
| **Average Holding Period (Trading Days)** | 7.20 days |
| **Maximum Holding Period (Trading Days)** | 26 days |
| **Minimum Holding Period (Trading Days)** | 1 day |
| **Average Holding Period (Calendar Days)** | 11.10 days |
| **Maximum Holding Period (Calendar Days)** | 41 days |
| **Minimum Holding Period (Calendar Days)** | 1 day |
| **Longest Continuous Position** | 26 days |
| **Percentage of Time Invested** | 12.10% |
| **Flat Market Percentage** | 87.90% |
| **Average Z-Score at Entry (Trigger)** | -2.3077 |
| **Average Z-Score at Entry (Execution)** | -1.8387 |
| **Average Z-Score at Exit (Trigger)** | -0.1867 |
| **Average Z-Score at Exit (Execution)** | -0.1220 |

*Note: The difference between trigger Z-score and execution Z-score represents Z-score slippage due to the 1-day lag necessary to prevent look-ahead bias.*

---

## 4. Parameter Sensitivity Analysis

A parameter grid search was performed over windows ($10, 20, 50$) and entry thresholds ($-1.5, -2.0, -2.5$) with a fixed exit threshold of $-0.5$.

| Window ($W$) | Entry Threshold ($ET$) | Exit Threshold | Number of Trades | Avg Hold (Trading Days) | Pct Time Invested |
|---|---|---|---|---|---|
| 10 | -1.5 | -0.5 | 128 | 4.67 | 20.40% |
| 10 | -2.0 | -0.5 | 66 | 4.92 | 11.10% |
| 10 | -2.5 | -0.5 | 2 | 6.50 | 0.44% |
| **20** | **-1.5** | **-0.5** | **85** | **7.35** | **21.39%** |
| **20 (Default)** | **-2.0** | **-0.5** | **49** | **7.20** | **12.10%** |
| **20** | **-2.5** | **-0.5** | **18** | **8.39** | **5.16%** |
| 50 | -1.5 | -0.5 | 36 | 14.31 | 17.82% |
| 50 | -2.0 | -0.5 | 25 | 16.88 | 14.57% |
| 50 | -2.5 | -0.5 | 14 | 18.86 | 9.12% |

### Key Sensitivities Observed
1. **Window Size Influence**: Increasing the window size $W$ leads to fewer trades but longer holding periods. A larger window computes rolling mean and std over a larger sample, making it harder for local price movements to trigger entry, and once triggered, the slow mean adjustment increases the time required to revert to $-0.5$.
2. **Entry Threshold Influence**: Tightening the entry threshold (making it more negative, e.g., $-1.5 \to -2.5$) decreases the trade count because extreme deviations are less frequent. For $W=10$, tightening to $-2.5$ drops the trade count to just 2, showing that short-term volatility standard deviations rarely expand to $2.5$ standard deviations without being absorbed by a fast-updating 10-day mean.

---

## 5. Strategic Evaluation

### Advantages
* **High Statistical Rigor**: Using Z-Scores instead of raw price channels standardizes signals, making them robust across regimes and price levels.
* **Low Time in Market**: At a $12.1\%$ market exposure, the strategy keeps the portfolio in cash $87.9\%$ of the time, reducing exposure to market tail risk.
* **FSM Control**: Hysteresis via different entry and exit thresholds prevents rapid buy-sell churning.

### Disadvantages
* **Negative Skew (Fat Tails)**: While mean reversion has high win rates in range-bound markets, it can face catastrophic losses if a position is entered during a structural sell-off (e.g., COVID-19 crash in March 2020) and the price fails to revert.
* **Execution Slippage**: As shown by the $0.47$ Z-Score difference between entry trigger ($-2.31$) and execution ($-1.84$), executing the next day causes significant decay in signal strength.

### Known Limitations
* **Lookback Sensitivity**: Results vary heavily depending on the choice of window.
* **No Stop-Loss**: The FSM is purely threshold-driven and lacks risk-management overrides for tail events.

### Possible Improvements
1. **Integrate a Trend Filter**: Use a macro indicator (like a 200-day SMA) to disable mean-reversion buying during structural downtrends.
2. **Dynamic Exit (Stop-Loss)**: Add a time-based exit or a protective stop-loss (e.g., exit if Z-Score goes below $-3.5$).
3. **Volatility-Adjusted Thresholds**: Scale entry/exit thresholds dynamically using historical volatility regimes (e.g. VIX).
