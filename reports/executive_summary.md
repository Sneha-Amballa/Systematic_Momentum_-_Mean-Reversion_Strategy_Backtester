# Executive Summary: Systematic Strategy Backtester & Regime Evaluation

**Date**: August 2026  
**Author**: Quantitative Research Group  
**Target Audience**: Portfolio Managers & Risk Committees  

---

## 1. Problem Statement & Research Objective

Many quantitative strategies fail in production because of **look-ahead leakage**, **curve-fitting (parameter overfitting)**, and **market regime shifts**. 

This research project builds a modular, production-ready quantitative research platform to backtest, validate, and analyze two standard trading models—**Dual SMA Momentum Crossover** and **Rolling Z-Score Mean Reversion**—on daily pricing data for the Nifty 50 index (2013-2024).

Our primary research objective is to determine:
1. Which strategy demonstrates a consistent statistical edge under realistic transaction costs (5.0 bps fee + 2.0 bps slippage on entry and exit legs).
2. Whether performance generalizes successfully to unseen out-of-sample regimes (2020-2024).
3. The specific market environments (Regimes) where each strategy exhibits an edge or underperforms.

---

## 2. Core Quantitative Methodology

* **Chronological splits**: We enforce a strict split to prevent look-ahead bias:
  * **In-Sample (IS)**: 2013-01-01 to 2019-12-31 (Used for parameter grid optimization).
  * **Out-of-Sample (OOS)**: 2020-01-01 to 2024-12-31 (Locked for final validation).
* **Execution & Friction Models**: Trades execute using next-day `open` or `close` prices. The engine accounts for transaction costs and slippage on all entry and exit legs.
* **Sensitivity Analysis**: We evaluate neighboring parameters in-sample (e.g. lookback window offsets) to compute parameter stability scores.
* **Regime Classification**: Daily prices are classified into 11 distinct states using combinations of moving-average trends, rolling realized volatility quantiles (expanding window), and crash/recovery flags.

---

## 3. Major Findings & Performance Scorecard

The table below summarizes performance across the full historical dataset (2013-2024) using the parameters optimized during the In-Sample phase:

| Metric | Momentum Crossover (`10/150`) | Mean Reversion Z-Score (`30/-2.0/-0.5`) | Benchmark (Buy & Hold) |
|---|---|---|---|
| **CAGR (%)** | 9.94% | 2.50% | 12.82% |
| **Annual Volatility (%)** | 10.88% | 11.23% | 15.65% |
| **Sharpe Ratio** | 0.9136 | 0.2227 | 0.8192 |
| **Sortino Ratio** | 1.3051 | 0.3168 | 1.1542 |
| **Max Drawdown (%)** | 22.34% | 34.78% | 38.44% |
| **Calmar Ratio** | 0.4449 | 0.0719 | 0.3335 |
| **Total Trade Cycles** | 15 | 40 | N/A |
| **Average Hold (Days)** | 134.5 | 11.2 | N/A |
| **Win Rate (%)** | 53.33% | 80.00% | N/A |

### Validation Results (In-Sample vs. Out-of-Sample)
* **Momentum**: Gen Score = **100.00%**, Stability Score = **95.40%**. Sharpe ratio improved from **0.77** (IS) to **1.08** (OOS).
* **Mean Reversion**: Gen Score = **3.29%**, Stability Score = **79.89%**. Sharpe ratio collapsed from **0.64** (IS) to **0.02** (OOS).

---

## 4. Key Insights & Strategy Edges

1. **Momentum generalizability**: The Momentum Crossover strategy is robust. Because it capitalizes on broad macroeconomic trends, its performance is stable across parameter neighborhoods and actually improved during Nifty's massive post-COVID bull market.
2. **Mean Reversion tail risk**: The Mean Reversion strategy is fragile. While it achieves a high daily win rate (**80%**), it carries severe tail risk. During a **Crash** regime (e.g. March 2020), it buys early on standard deviations but gets trapped holding a falling index, resulting in a **34.78% maximum drawdown**. Lacking a stop-loss, it remains locked in losing trades for years, paralyzing capital.
3. **Complementary Regime Profiles**: Momentum thrives in trending low-volatility expansion markets, whereas Mean Reversion has a strong edge during high-volatility expansions (buying quick dips in a bull market, Sharpe: **2.68**).

---

## 5. Strategic Portfolio Recommendations

1. **Deploy Momentum with Confidence**: The 10/150 Momentum strategy has proven its generalizability and is ready for production staging.
2. **Reject Unhedged Mean Reversion**: Discard the Z-Score strategy in its current form. It requires the addition of a hard volatility filter (deactivating entries when VIX > 25) and a strict stop-loss.
3. **Implement Regime-Switching Allocation**: Capital should be dynamically allocated between momentum and mean-reversion based on rolling volatility and trend indicators.
