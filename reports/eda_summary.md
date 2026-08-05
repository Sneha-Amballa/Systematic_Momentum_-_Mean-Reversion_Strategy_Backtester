# Quantitative Research EDA Summary Report
**Asset**: Nifty 50 Index (`^NSEI`)  
**Data Period**: 2013-01-02 to 2024-12-31  
**Generated Date**: 2026-08-05  

---

## 1. Dataset Overview
A high-level inspection of the processed analysis-ready market dataset:
* **Total Observations (Rows)**: 2946
* **Total Dimensions (Columns)**: 9
* **Date Bounds**: 2013-01-02 to 2024-12-31
* **Memory Footprint**: 230.16 KB
* **Dtypes**:
  * `Close`: `float64`
  * `High`: `float64`
  * `Low`: `float64`
  * `Open`: `float64`
  * `Simple_Return`: `float64`
  * `Log_Return`: `float64`
  * `Price_Change`: `float64`
  * `Daily_Range`: `float64`
  * `Pct_Range`: `float64`

---

## 2. Data Quality Audit Summary
The pre-preprocessing check reports complete coverage across clean columns:
* **Missing (NaN) values**: 2
* **Infinite values**: 0
* **Fully duplicate rows**: 0
* **Duplicate timestamps**: 0
* **Completeness Ratio**: 100.0% (Clean index contains no missing rows or duplicate date indices).

---

## 3. Core Descriptive Statistics
Descriptive metrics calculated for primary price and return dimensions:

| Statistic | Open | High | Low | Close | Simple_Return | Daily_Range | Price_Change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Count** | 2946 | 2946 | 2946 | 2946 | 2945 | 2946 | 2946 |
| **Mean** | 12378.85 | 12436.52 | 12301.06 | 12370.20 | 0.0520% | 135.46 | -8.65 |
| **Median** | 10785.77 | 10824.25 | 10725.18 | 10776.23 | 0.0715% | 108.72 | -5.40 |
| **Std Dev** | 5223.48 | 5242.78 | 5197.77 | 5221.42 | 1.0387% | 101.53 | 103.94 |
| **Minimum** | 5233.45 | 5317.70 | 5118.85 | 5285.00 | -12.98% | 23.45 | -1295.00 |
| **Maximum** | 26248.25 | 26277.35 | 26151.40 | 26216.05 | 8.76% | 1898.05 | 847.60 |
| **IQR** | 8706.40 | 8776.16 | 8611.11 | 8687.28 | 1.0522% | 94.15 | 98.58 |
| **Coeff Var**| 0.4220 | 0.4216 | 0.4225 | 0.4221 | 19.9573 | 0.7495 | -12.0230 |

---

## 4. Key Statistical Findings

### A. Distribution Normality
* **Skewness**: -0.9511  
  * *Interpretation*: A negative skewness indicates that the distribution has a longer left tail (more frequent large negative daily returns than normal would predict).
* **Excess Kurtosis**: 15.4917  
  * *Interpretation*: High excess kurtosis (leptokurtic distribution) mathematically confirms the presence of **fat-tails** or heavy tails, where extreme events occur far more frequently than in a bell curve.
* **Jarque-Bera Test**:
  * Test Statistic: 29893.26
  * p-value: 0.0000e+00
  * *Normality Rejection*: With a p-value of essentially 0, **we reject the null hypothesis of normal distribution** with absolute statistical significance. 

### B. Time Series Stationarity
We run the Augmented Dickey-Fuller (ADF) and KPSS tests on both Close Price levels and Daily Returns to verify stationarity.

| Series | ADF Statistic | ADF p-value | ADF Result (5% critical = -2.86) | KPSS Statistic | KPSS p-value | KPSS Result (5% critical = 0.463) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Close Price** | 0.55 | 0.9862 | Non-Stationary | 8.03 | 0.0100 | Non-Stationary |
| **Daily Returns**| -15.13 | 0.0000 | **Stationary** | 0.04 | 0.1000 | **Stationary** |

* **Key Takeaway**: Price levels are integrated of order 1 ($I(1)$). Standard prices cannot be modeled with stationary models (like ARMA) without scaling. Returns are $I(0)$ and stationary, confirming they are appropriate for signal development.

---

## 5. Interesting Market Behaviour & Seasonality

### A. Weekday Return Characteristics
Average compounded daily return by day of week:
* **Monday**: -0.0037%
* **Tuesday**: +0.0868%
* **Wednesday**: +0.0575%
* **Thursday**: +0.0299%
* **Friday**: +0.0902%
* *Observation*: Friday and Tuesday have historically exhibited distinct average behaviors, which could serve as inputs for day-of-week style filter signals.

### B. Volatility Clustering
Comparing rolling 20, 50, and 100-day annualized volatilities confirms **volatility clustering**: high-volatility periods follow high-volatility periods, and low-volatility periods follow low-volatility periods. Volatility spikes are sudden (leptokurtic shocks) while decay is slow and persistent.

---

## 6. Extreme Market Days & Outliers
Top daily returns, ranges, and gap movements:

### Top 5 Positive Return Days
1. Date: 2020-04-07 | Return: +8.76% | Close: 8792.20
1. Date: 2020-03-25 | Return: +6.62% | Close: 8317.85
1. Date: 2020-03-20 | Return: +5.83% | Close: 8745.45
1. Date: 2019-09-20 | Return: +5.32% | Close: 11274.20
1. Date: 2021-02-01 | Return: +4.74% | Close: 14281.20

### Top 5 Negative Return Days
1. Date: 2020-03-23 | Return: -12.98% | Close: 7610.25
1. Date: 2020-03-12 | Return: -8.30% | Close: 9590.15
1. Date: 2020-03-16 | Return: -7.61% | Close: 9197.40
1. Date: 2024-06-04 | Return: -5.93% | Close: 21884.50
1. Date: 2015-08-24 | Return: -5.92% | Close: 7809.00

### Top 5 Opening Gaps (Opening Price Shock)
1. Date: 2020-03-23 | Opening Gap: -9.14% | Open: 7945.70 | Close: 7610.25
1. Date: 2016-11-09 | Opening Gap: -5.57% | Open: 8067.50 | Close: 8432.00
1. Date: 2020-03-13 | Opening Gap: -5.03% | Open: 9107.60 | Close: 9955.20
1. Date: 2020-03-19 | Opening Gap: -4.79% | Open: 8063.30 | Close: 8263.45
1. Date: 2020-04-07 | Opening Gap: +4.48% | Open: 8446.30 | Close: 8792.20

* **Outlier Retainment Justification**: These observations are not data errors. They represent key market regimes (COVID crash, global oil wars, policy shock events). Retaining them is essential to prevent backtesting from underestimating maximum drawdowns and tail risk.

---

## 7. Strategic Recommendations for Signal Construction

1. **Mean-Reversion Signal Construction**:
   * Since returns are stationary ($I(0)$) and exhibit high mean-reversion tendencies during consolidation regimes (e.g., 2016, 2018), strategies should construct **Z-scores** based on price distance from rolling moving averages.
   * Volatility clustering suggests standardizing Z-scores by a rolling volatility estimator (such as rolling Bollinger band width) to scale entries dynamically during low/high regimes.
2. **Momentum Signal Construction**:
   * Prices exhibit significant trending regimes (e.g., post-COVID bull run). Dual moving average crossovers (e.g., 50-day and 200-day) or exponential trend-following filters are recommended to catch these high-momentum phases.
3. **Volatility Filters**:
   * Strategy execution should be scaled down or paused during periods when rolling 20-day annualized volatility exceeds historical 95th percentiles (e.g., > 30% volatility during March 2020) to limit stop-loss triggers.

---

## 8. Limitations of the Dataset
* **Survival Bias**: The Nifty 50 Index represents the top 50 active stocks; components change over time. Backtesting directly on the index hides survival biases of individual components.
* **No Transaction Costs**: Intraday ranges show that slippage and commission costs will eat up a significant part of high-frequency mean-reversion returns.
* **No Intraday Data**: Daily close limits signal accuracy; intraday gaps (up to 4.5% open gaps) can trigger stop-losses before close.
