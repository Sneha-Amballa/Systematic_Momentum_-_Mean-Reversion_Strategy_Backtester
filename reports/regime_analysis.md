# Market Regime Analysis Report

This research report presents a comprehensive market regime analysis of the Nifty 50 index (2013-2024), analyzing how the **Momentum Crossover** and **Rolling Z-Score Mean Reversion** strategies perform in distinct market environments.

---

## 1. Executive Summary

Instead of evaluating strategies on a static, aggregate basis, we analyze performance across objective market regimes (Trend + Volatility + Event states). Our findings show that strategy returns are heavily dependent on these structural states:
* **Momentum Crossover** has a strong edge in **Trending Up** environments, achieving its highest annualized Sharpe ratios in **Trending Up Low Volatility (Sharpe: 1.65)** and **Trending Up High Volatility (Sharpe: 1.57)** states. However, it suffers massive whipsaw losses during **Trending Down (Sharpe: -3.07)** and **Sideways High Volatility (Sharpe: -1.06)** regimes.
* **Mean Reversion Z-Score** exhibits a strong edge during **Trending Up High Volatility (Sharpe: 2.68)** and **Recovery (Sharpe: 1.31)** regimes. However, it suffers catastrophic tail risk during a **Crash (Sharpe: -2.21, Max DD: 30.75%)** because it "buys the dip" into structural bear markets without a stop-loss.

---

## 2. Objective Regime Classification Methodology

We classified Nifty 50 daily behavior into 11 distinct combined regimes:
1. **Trend State**:
   * **Trending Up**: Index Close > 200-day SMA AND 50-day SMA > 200-day SMA.
   * **Trending Down**: Index Close < 200-day SMA AND 50-day SMA < 200-day SMA.
   * **Sideways**: Otherwise.
2. **Volatility State** (Low / Medium / High): Partitions 20-day realized volatility daily using expanding quantiles (33% and 66%) to prevent look-ahead bias.
3. **Event Overlays**:
   * **Crash**: Drawdown from index peak > 15% AND rolling 20-day return < -10%.
   * **Recovery**: Drawdown from index peak > 10% AND rolling 20-day return > 0.0%.

### Historical Occurrence Frequency (2013-2024)
The index spent the majority of its history in **Trending Up Low Volatility (1,029 days, 35.2%)** and **Trending Up Medium Volatility (527 days, 18.0%)**, representing a strong structural upward bias. The catastrophic **Crash** regime was highly transient, occurring on only **23 days (0.8%)**, while the **Recovery** state lasted **144 days (4.9%)**.

---

## 3. Comparative Performance by Regime

The table below summarizes performance metrics computed on synthetic contiguous return series for each strategy by regime.

### Momentum Crossover Strategy
| Regime | Active Days | CAGR (%) | Volatility (%) | Sharpe Ratio | Max Drawdown (%) | Exposure (%) |
|---|---|---|---|---|---|---|
| **Crash** | 23 | 0.00% | 0.00% | 0.0000 | 0.00% | 0.00% |
| **Recovery** | 144 | 6.74% | 12.10% | 0.5568 | 2.94% | 46.53% |
| **Sideways High Vol** | 288 | -13.06% | 12.48% | -1.0564 | 11.51% | 51.74% |
| **Sideways Low Vol** | 148 | 8.86% | 5.94% | 1.4921 | 2.44% | 59.46% |
| **Sideways Medium Vol** | 157 | 10.98% | 7.33% | 1.4983 | 2.85% | 77.07% |
| **Trending Down High Vol** | 131 | -17.43% | 11.75% | -1.4911 | 2.29% | 49.62% |
| **Trending Down Low Vol** | 65 | -15.54% | 5.06% | -3.0720 | 3.00% | 15.38% |
| **Trending Down Med Vol** | 53 | -14.65% | 6.03% | -2.4307 | 2.44% | 75.47% |
| **Trending Up High Vol** | 381 | 21.05% | 13.43% | 1.5680 | 10.03% | 69.82% |
| **Trending Up Low Vol** | 1029 | 15.69% | 9.53% | 1.6458 | 9.14% | 83.19% |
| **Trending Up Medium Vol** | 527 | 2.50% | 10.63% | 0.2350 | 22.34% | 73.06% |

### Mean Reversion Z-Score Strategy
| Regime | Active Days | CAGR (%) | Volatility (%) | Sharpe Ratio | Max Drawdown (%) | Exposure (%) |
|---|---|---|---|---|---|---|
| **Crash** | 23 | -44.57% | 23.47% | -2.2137 | 30.75% | 100.00% |
| **Recovery** | 144 | 16.53% | 12.65% | 1.3070 | 0.02% | 8.33% |
| **Sideways High Vol** | 288 | -1.33% | 12.33% | -0.1080 | 15.90% | 17.71% |
| **Sideways Low Vol** | 148 | -7.21% | 5.65% | -1.2748 | 1.70% | 2.03% |
| **Sideways Medium Vol** | 157 | 0.07% | 7.37% | 0.0089 | 6.79% | 14.01% |
| **Trending Down High Vol** | 131 | 11.23% | 12.26% | 0.9162 | 6.59% | 22.14% |
| **Trending Down Low Vol** | 65 | 9.47% | 5.06% | 1.8705 | 1.35% | 40.00% |
| **Trending Down Med Vol** | 53 | -7.53% | 5.99% | -1.2577 | 4.47% | 5.66% |
| **Trending Up High Vol** | 381 | 37.33% | 13.91% | 2.6843 | 4.09% | 29.13% |
| **Trending Up Low Vol** | 1029 | 1.10% | 9.53% | 0.1158 | 3.39% | 1.17% |
| **Trending Up Medium Vol** | 527 | 12.44% | 10.61% | 1.1721 | 7.19% | 14.80% |

---

## 4. Regime Stability & Transition Matrix

### Stability Analysis
* **Transition Frequency**: We identified **359 regime transitions** over 12 years.
* **Persistence**: Regimes are relatively short-lived, with an **average duration of 8.18 trading days**. 
* **Longest Regime**: **Trending Up Low Volatility** has the highest persistence, with contiguous blocks lasting over **100 days** during major multi-year expansions.

### Markov Transition Probability Matrix (Selected Rows)
* **Crash State Probability**: If in a **Crash** state today, there is a **91.3% probability** of remaining in a Crash state tomorrow, and an **8.7% probability** of transitioning to **Recovery**.
* **Trending Up Low Vol Probability**: If in **Trending Up Low Volatility** today, there is a **95.5% probability** of remaining there tomorrow, and a **4.5% probability** of shifting to **Trending Up Medium Volatility**.
* **Sideways High Vol**: High probability of transitioning to **Trending Down High Volatility (8.3%)** or **Trending Up High Volatility (11.5%)**, indicating that high-volatility sideways markets are highly unstable transition nodes preceding breakout trends.

---

## 5. Feature Importance Correlation

We calculated the absolute Pearson correlation coefficient between key market factors and strategy returns:
1. **Index Daily Return**: **0.6785** (Highest importance - strategy returns are strongly driven by daily market direction).
2. **Index Rolling Return (20d)**: **0.1628**
3. **Index Drawdown**: **0.1257**
4. **Trend Strength**: **0.0277**
5. **Index Volatility**: **0.0086**

---

## 6. Key Findings and Limitations

### Key Findings
* **Regime Divergence**: Momentum and Mean Reversion have highly complementary regime profiles. Momentum performs best in low-volatility trending markets, while Mean Reversion performs best in high-volatility bull markets (buying temporary pullbacks during an expansion).
* **Mean Reversion Tail Risk**: The Mean Reversion strategy's low performance is concentrated in the **Crash** regime (Sharpe: -2.21, Max DD: 30.75%). If this regime could be filtered out using a VIX filter or a stop-loss, the strategy would generate substantial alpha.

### Limitations
* **Lag in SMA Transitions**: MA crossover signals lag behind sudden regime changes (e.g. from Trending Up to Crash), meaning the strategy remains exposed to the initial, most violent days of market sell-offs.
* **Transition Return Horizon**: Transition analysis is backward-looking and assumes that future regime transitions will follow the historical transition probability distribution.
