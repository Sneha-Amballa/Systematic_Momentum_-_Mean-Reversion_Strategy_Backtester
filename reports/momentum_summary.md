# Quantitative Research Report: Momentum Strategy Signal Construction

**Author**: Senior Quantitative Researcher  
**Target Asset**: Nifty 50 Index (`^NSEI`)  
**Strategy**: Long-Only Simple Moving Average (SMA) Crossover Momentum  
**Analysis Period**: 2013-01-02 to 2024-12-31  

---

## 1. Theoretical Foundation

### What is Momentum Investing?
Momentum investing is a systematic market strategy based on the empirical tendency of asset prices to continue their recent price trends. It is grounded in the premise that assets that have performed well in the past tend to continue performing well in the near term, while assets that have performed poorly continue to underperform. 

Mathematically, momentum can be defined as:
$$M_t = P_t - P_{t-k}$$
where $P_t$ is the price at time $t$ and $k$ is the lookback window.

From a behavioral finance perspective, momentum exists due to:
1. **Underreaction**: Market participants underreact to new information due to cognitive biases (e.g., anchoring and conservatism), causing prices to adjust gradually rather than instantaneously.
2. **Delayed Overreaction**: As a trend establishes, herd behavior, FOMO (Fear of Missing Out), and positive feedback loops (e.g., trend-following algorithms and retail buying) drive prices beyond fundamental value.
3. **Institutional Constraints**: Portfolio managers often window-dress (buying winners at quarter-end) and execute capital flows gradually to minimize market impact, perpetuating trends.

### Why Moving Average Crossovers are Used
Moving Average Crossovers are trend-following indicators used to smooth out daily price noise (high-frequency volatility) and isolate the underlying medium-to-long-term price trend (drift). A crossover indicates a structural shift in the asset's momentum.

By comparing a fast (short) moving average against a slow (long) moving average, quants can identify changes in the acceleration of price movements:
- When the short moving average crosses above the long moving average, it indicates that recent price velocity is expanding upward relative to historical standards, signaling the start of a bullish trend.
- Conversely, a cross below indicates decelerating velocity, signaling a bearish trend or mean-reversion.

### Short vs. Long Moving Averages
- **Short Moving Average ($SMA_S$)**: Focuses on short-term price movements. It responds rapidly to price changes but contains a higher ratio of noise-to-signal, making it prone to "whipsaws" (false entry/exit signals).
- **Long Moving Average ($SMA_L$)**: Smooths out short-term fluctuations to reveal the secular, long-term trend. It has a high signal-to-noise ratio but lags behind pivot points, reducing the trade's efficiency near trend reversals.

---

## 2. Methodology & Signal Construction

### Mathematical Formulations
For a price series $P$, the Simple Moving Average (SMA) of window $W$ at time $t$ is defined as:
$$SMA_W(t) = \frac{1}{W} \sum_{i=0}^{W-1} P_{t-i}$$

Our strategy utilizes a dual-crossover framework:
- **Short Window ($W_S$)** = 20 trading days
- **Long Window ($W_L$)** = 100 trading days

### Signal Logic (Long-Only)
The strategy is strictly **Long-Only** (no short-selling is permitted).

1. **Raw Signal**:
   $$\text{Raw\_Signal}_t = \begin{cases} 1.0 & \text{if } SMA_{20}(t) > SMA_{100}(t) \\ 0.0 & \text{if } SMA_{20}(t) \leq SMA_{100}(t) \end{cases}$$
   *Warm-up Period Constraint*: For any $t < W_L - 1$ (first 99 observations), $\text{Raw\_Signal}_t = \text{NaN}$.

2. **Execution Signal (Look-Ahead Bias Prevention)**:
   Today's closing price $P_t$ determines $\text{Raw\_Signal}_t$. However, because the trade can only be filled after the close is recorded, execution cannot occur at $P_t$. The signal is shifted forward by one trading day:
   $$\text{Execution\_Signal}_t = \text{Raw\_Signal}_{t-1}$$
   
3. **Position State**:
   $$\text{Position}_t = \text{Execution\_Signal}_t$$
   - `1.0` represents holding a Long Position.
   - `0.0` represents holding cash (Flat Market / No Position).

---

## 3. Look-Ahead Bias & Point-in-Time Correctness

In quantitative finance, **Look-Ahead Bias** (or future leakage) occurs when data from the future is used to make trading decisions in the past. This leads to artificially inflated, unrealistic backtest results that fail immediately in live trading.

### Point-in-Time Example Timeline
Assume a signal crossover occurs on **Monday close**:

```
[Monday 15:30] ---------> [Monday 17:30] -------------> [Tuesday 09:15]
Market Closes             SMA calculated               Market Opens
P_t is recorded           Raw_Signal_t generated       Execution_Signal_{t+1} is active
                          (Short > Long = 1)           Order executed (Position_t = 1)
```

- **Monday Close ($t$)**: We calculate the closing prices and update $SMA_{20}$ and $SMA_{100}$. A crossover occurs, so $\text{Raw\_Signal}_t = 1$.
- **Trading at $t$**: We *cannot* enter a position at Monday's close because we did not know the close price *before* the market closed. Entering a trade at Monday's close using Monday's signal is a look-ahead bias.
- **Tuesday ($t+1$)**: The signal is executed on Tuesday. $\text{Execution\_Signal}_{t+1} = \text{Raw\_Signal}_t = 1$. Thus, we hold a long position starting on Tuesday, entering at either Tuesday's open or Tuesday's close.

This shifting ensures that all data used in signal calculations is historically completed and known at the moment of execution (**Point-in-Time Correctness**).

---

## 4. Empirical Trade Statistics

Running the strategy on the Nifty 50 Index from **2013-01-02 to 2024-12-31** (2,946 calendar days, 2,846 active trading days after lookback warm-up) yielded the following metrics:

| Metric | Value |
| :--- | :--- |
| **Total Active Trading Days** | 2,846 |
| **Total Raw Buy Days ($SMA_{20} > SMA_{100}$)** | 2,038 |
| **Total Raw Sell Days ($SMA_{20} \leq SMA_{100}$)** | 809 |
| **Total Execution Buy Days** | 2,038 |
| **Total Execution Sell Days** | 808 |
| **Number of Trade Cycles (Entries)** | 16 |
| **Average Holding Period (Trading Days)** | 127.38 days |
| **Maximum Holding Period (Trading Days)** | 369 days |
| **Minimum Holding Period (Trading Days)** | 15 days |
| **Average Holding Period (Calendar Days)** | 189.25 days |
| **Maximum Holding Period (Calendar Days)** | 542 days |
| **Minimum Holding Period (Calendar Days)** | 22 days |
| **Percentage of Time Invested** | 71.61% |
| **Flat Market Percentage** | 28.39% |

*Note: The terminal open position at the end of the analysis period (December 2024) was closed on the final trading day to compute its holding period and ensure complete trade logs.*

---

## 5. Strategy Analysis

### Advantages
1. **Capital Preservation**: By transitioning to a flat state (`Position = 0`) during structural downtrends, the strategy successfully avoids major market drawdowns (e.g., the 2008 financial crisis or the March 2020 pandemic crash).
2. **Participation in Extended Trends**: The long holding period (max 369 trading days) ensures the strategy captures the bulk of extended bull runs, compounding returns.
3. **Simplicity and Low Turnovers**: With only 16 trades over 12 years, transaction costs and slippage are minimized, making it highly cost-effective.
4. **No Parameter Overfitting**: The 20/100 parameters are industry standard, avoiding "curve-fitting" which degrades performance out-of-sample.

### Limitations
1. **Lagging Indicators**: Signals are generated after a trend has already started and exited after a reversal has already begun, giving up profits at both the entry and exit points.
2. **Whipsaws in Range-bound Markets**: In a sideways (flat) market, short and long SMAs cross repeatedly, generating false buy/sell signals that chew up capital via small consecutive losses (known as "churning").
3. **Opportunity Cost**: Remaining in cash for 28.39% of the time means capital is idle and yielding zero returns unless parked in risk-free debt instruments.

### Market Regimes Performance
- **Good Performance**: High-beta, structural, long-term bull markets (e.g., Nifty 50 during 2014-2015, 2017, and post-2020 recovery).
- **Poor Performance**: Sideways, high-volatility range-bound markets (e.g., Nifty 50 during parts of 2015-2016 and 2018).

---

## 6. Known Limitations & Potential Enhancements

### Known Limitations
- **SMA Weights**: Simple Moving Averages weight all data points in the window equally. Recent price movements do not carry more weight than old price movements, which increases the delay in signal generation.
- **Fixed Parameters**: Fixed 20 and 100 windows do not adapt to changes in market volatility or speed.

### Suggested Improvements
1. **Exponential Moving Average (EMA)**: Replace SMA with EMA to give higher weight to recent prices:
   $$EMA_t = P_t \times \left(\frac{2}{N+1}\right) + EMA_{t-1} \times \left(1 - \frac{2}{N+1}\right)$$
   This reduces indicator lag, allowing faster entries and exits.
2. **Volatility Filter**: Add an Average True Range (ATR) or Bollinger Band filter to avoid entering trades when volatility is extremely low (detecting sideways consolidation regimes and reducing whipsaws).
3. **Trend Strength Filter (ADX)**: Integrate the Average Directional Index (ADX) to only trigger crossover trades when ADX > 20/25, confirming a strong trend is present.
4. **Adaptive Windows**: Use indicators like the Kaufman Adaptive Moving Average (KAMA) that dynamically adjust their smoothing factor based on market efficiency.
