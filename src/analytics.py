"""
Strategy Analytics and Performance Dashboard Module.

Computes trade book statistics, benchmark capture ratios, rolling metrics,
drawdown event analysis, scorecards, and generates 20 publication-quality charts.
"""

import os
import logging
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from src.risk import calculate_volatility, calculate_downside_volatility, calculate_maximum_drawdown, calculate_drawdown_series
from src.metrics import calculate_return_metrics, calculate_risk_adjusted_ratios, calculate_distribution_metrics

logger = logging.getLogger(__name__)

def calculate_trade_statistics(trade_book: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes trade-level performance metrics from the Trade Book.

    Args:
        trade_book (pd.DataFrame): Trade book log from backtest results.

    Returns:
        Dict[str, Any]: Dictionary of trade statistics.
    """
    total_trades = len(trade_book)
    if total_trades == 0:
        return {
            "Total_Trades": 0, "Winning_Trades": 0, "Losing_Trades": 0,
            "Win_Rate": 0.0, "Loss_Rate": 0.0, "Average_Profit": 0.0, "Average_Loss": 0.0,
            "Profit_Factor": 0.0, "Payoff_Ratio": 0.0, "Expectancy": 0.0,
            "Largest_Winner": 0.0, "Largest_Loser": 0.0, "Average_Hold_Days": 0.0,
            "Max_Hold_Days": 0, "Min_Hold_Days": 0, "Average_Trade_Return": 0.0, "Trade_Frequency": 0.0
        }

    # Net trade returns (gross return - total cost return)
    # Return of a trade = size * (exit_price - entry_price) / entry_price
    gross_returns = trade_book["Position Size"] * (trade_book["Exit Price"] - trade_book["Entry Price"]) / trade_book["Entry Price"]
    
    # We estimate cost in return terms: Total Cost / (100000.0 * |Position Size|)
    # Since initial_capital is ₹100,000, the total cost divided by value is the return drag
    cost_returns = trade_book["Total Cost"] / (100000.0 * trade_book["Position Size"].abs())
    net_trade_returns = gross_returns - cost_returns

    winners = net_trade_returns[net_trade_returns > 0.0]
    losers = net_trade_returns[net_trade_returns <= 0.0]

    win_count = len(winners)
    loss_count = len(losers)
    win_rate = win_count / total_trades
    loss_rate = loss_count / total_trades

    avg_profit = float(winners.mean()) if win_count > 0 else 0.0
    avg_loss = float(losers.mean()) if loss_count > 0 else 0.0

    sum_profit = float(winners.sum()) if win_count > 0 else 0.0
    sum_loss = float(losers.sum()) if loss_count > 0 else 0.0
    profit_factor = sum_profit / abs(sum_loss) if sum_loss != 0.0 else np.nan

    payoff_ratio = avg_profit / abs(avg_loss) if avg_loss != 0.0 else np.nan
    expectancy = (win_rate * avg_profit) - (loss_rate * abs(avg_loss))

    largest_winner = float(net_trade_returns.max())
    largest_loser = float(net_trade_returns.min())

    avg_hold = float(trade_book["Holding Days"].mean())
    max_hold = int(trade_book["Holding Days"].max())
    min_hold = int(trade_book["Holding Days"].min())

    avg_trade_return = float(net_trade_returns.mean())

    return {
        "Total_Trades": total_trades,
        "Winning_Trades": win_count,
        "Losing_Trades": loss_count,
        "Win_Rate": win_rate * 100.0,
        "Loss_Rate": loss_rate * 100.0,
        "Average_Profit_Pct": avg_profit * 100.0,
        "Average_Loss_Pct": avg_loss * 100.0,
        "Profit_Factor": profit_factor,
        "Payoff_Ratio": payoff_ratio,
        "Expectancy_Pct": expectancy * 100.0,
        "Largest_Winner_Pct": largest_winner * 100.0,
        "Largest_Loser_Pct": largest_loser * 100.0,
        "Average_Holding_Period": avg_hold,
        "Maximum_Holding_Period": max_hold,
        "Minimum_Holding_Period": min_hold,
        "Average_Trade_Return_Pct": avg_trade_return * 100.0
    }

def calculate_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    annualization_factor: float = 252.0
) -> Dict[str, Any]:
    """
    Calculates metrics relative to a market benchmark.

    Args:
        strategy_returns (pd.Series): Strategy daily returns.
        benchmark_returns (pd.Series): Benchmark daily returns.
        risk_free_rate_annual (float): Annual risk-free rate. Default is 0.0.
        annualization_factor (float): Daily periods in a year. Default is 252.0.

    Returns:
        Dict[str, Any]: Benchmark analytics comparison dictionary.
    """
    if strategy_returns.empty or benchmark_returns.empty:
        return {}

    # Align series
    aligned_df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    strat_ret = aligned_df.iloc[:, 0]
    bench_ret = aligned_df.iloc[:, 1]

    cov = np.cov(strat_ret, bench_ret)[0, 1]
    bench_var = np.var(bench_ret, ddof=1)

    beta = cov / bench_var if bench_var > 0.0 else 0.0

    # Annual returns
    ann_strat_return = strat_ret.mean() * annualization_factor
    ann_bench_return = bench_ret.mean() * annualization_factor

    # Alpha: (Strat_Ann - Rf) - Beta * (Bench_Ann - Rf)
    alpha = (ann_strat_return - risk_free_rate_annual) - beta * (ann_bench_return - risk_free_rate_annual)

    # Tracking Error
    excess_daily = strat_ret - bench_ret
    tracking_error = excess_daily.std(ddof=1) * np.sqrt(annualization_factor)

    # Excess return
    excess_return = ann_strat_return - ann_bench_return

    correlation = float(strat_ret.corr(bench_ret))

    # Up/Down capture ratios
    up_days = bench_ret > 0.0
    down_days = bench_ret < 0.0

    up_capture = (strat_ret[up_days].mean() / bench_ret[up_days].mean()) * 100.0 if up_days.any() else np.nan
    down_capture = (strat_ret[down_days].mean() / bench_ret[down_days].mean()) * 100.0 if down_days.any() else np.nan

    return {
        "Beta": beta,
        "Alpha": alpha,
        "Tracking_Error": tracking_error,
        "Excess_Return": excess_return,
        "Up_Capture": up_capture,
        "Down_Capture": down_capture,
        "Correlation": correlation
    }

def calculate_rolling_metrics(
    portfolio_df: pd.DataFrame,
    window: int = 126,
    annualization_factor: float = 252.0
) -> pd.DataFrame:
    """
    Computes rolling performance metrics.

    Args:
        portfolio_df (pd.DataFrame): Backtest portfolio statistics.
        window (int): Size of rolling window in trading days. Default is 126.
        annualization_factor (float): Daily periods in a year.

    Returns:
        pd.DataFrame: DataFrame containing rolling metrics.
    """
    df = pd.DataFrame(index=portfolio_df.index)
    daily_returns = portfolio_df["Daily_Return"]

    # Rolling Volatility
    df["Rolling_Volatility"] = daily_returns.rolling(window).std(ddof=1) * np.sqrt(annualization_factor)

    # Rolling Sharpe (assuming RF = 0)
    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std(ddof=1)
    df["Rolling_Sharpe"] = (rolling_mean / rolling_std) * np.sqrt(annualization_factor)

    # Rolling CAGR
    val = portfolio_df["Portfolio_Value"]
    df["Rolling_CAGR"] = (val / val.shift(window)) ** (annualization_factor / window) - 1.0

    # Rolling Max Drawdown
    # Peak over rolling window
    rolling_peaks = val.rolling(window).max()
    rolling_drawdowns = (rolling_peaks - val) / rolling_peaks
    df["Rolling_Max_Drawdown"] = rolling_drawdowns.rolling(window).max()

    # Rolling Win Rate
    df["Rolling_Win_Rate"] = daily_returns.rolling(window).apply(lambda x: (x > 0.0).sum() / len(x)) * 100.0

    return df

def analyze_drawdowns(portfolio_value: pd.Series) -> pd.DataFrame:
    """
    Analyzes and lists every drawdown event chronologically.

    Args:
        portfolio_value (pd.Series): Portfolio value series.

    Returns:
        pd.DataFrame: Drawdown events log.
    """
    peaks = portfolio_value.cummax()
    in_drawdown = portfolio_value < peaks

    events = []
    in_dd_state = False
    peak_val = -1.0
    peak_date = None
    trough_val = -1.0
    trough_date = None
    start_date = None

    for i in range(len(portfolio_value)):
        val = portfolio_value.iloc[i]
        date = portfolio_value.index[i]

        if not in_dd_state:
            if val < peak_val:
                in_dd_state = True
                start_date = date
                trough_val = val
                trough_date = date
            else:
                peak_val = val
                peak_date = date
        else:
            if val >= peak_val:
                events.append({
                    "Start Date": start_date,
                    "Peak Date": peak_date,
                    "Trough Date": trough_date,
                    "Recovery Date": date,
                    "Depth (%)": ((peak_val - trough_val) / peak_val) * 100.0,
                    "Duration (TD)": len(portfolio_value.loc[start_date:date]) - 1,
                    "Recovered": True
                })
                in_dd_state = False
                peak_val = val
                peak_date = date
            else:
                if val < trough_val:
                    trough_val = val
                    trough_date = date

    if in_dd_state:
        final_date = portfolio_value.index[-1]
        events.append({
            "Start Date": start_date,
            "Peak Date": peak_date,
            "Trough Date": trough_date,
            "Recovery Date": pd.Timestamp("NaT"),
            "Depth (%)": ((peak_val - trough_val) / peak_val) * 100.0,
            "Duration (TD)": len(portfolio_value.loc[start_date:final_date]) - 1,
            "Recovered": False
        })

    df = pd.DataFrame(events)
    if df.empty:
        df = pd.DataFrame(columns=["Start Date", "Peak Date", "Trough Date", "Recovery Date", "Depth (%)", "Duration (TD)", "Recovered"])
    else:
        df = df.sort_values(by="Depth (%)", ascending=True)  # Depth is positive, sort ascending means shallowest first? No, we want deepest worst drawdown first. Depth is positive value, so sort descending.
        df = df.sort_values(by="Depth (%)", ascending=False)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df

def generate_scorecard(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Generates a performance scorecard with institutional qualitative ranking.

    Args:
        metrics (Dict[str, Any]): Dictionary of calculated metrics.

    Returns:
        pd.DataFrame: Scorecard summary table.
    """
    # Define metric scorecard evaluation thresholds
    thresholds = {
        "Sharpe_Ratio": {"Excellent": 1.5, "Good": 1.0, "Average": 0.5},
        "Sortino_Ratio": {"Excellent": 2.0, "Good": 1.5, "Average": 0.75},
        "Calmar_Ratio": {"Excellent": 1.5, "Good": 1.0, "Average": 0.5},
        "Win_Rate": {"Excellent": 60.0, "Good": 50.0, "Average": 40.0},
        "Profit_Factor": {"Excellent": 2.0, "Good": 1.5, "Average": 1.0},
        "Max_Drawdown": {"Excellent": 10.0, "Good": 20.0, "Average": 35.0, "invert": True}
    }

    scorecard_rows = []

    for metric_name, rules in thresholds.items():
        val = metrics.get(metric_name)
        if val is None or pd.isna(val):
            rank = "N/A"
        else:
            invert = rules.get("invert", False)
            if not invert:
                if val >= rules["Excellent"]:
                    rank = "Excellent"
                elif val >= rules["Good"]:
                    rank = "Good"
                elif val >= rules["Average"]:
                    rank = "Average"
                else:
                    rank = "Poor"
            else:
                # Lower is better (e.g. Drawdown)
                # Convert DD to positive if needed
                val_abs = abs(val)
                if val_abs <= rules["Excellent"]:
                    rank = "Excellent"
                elif val_abs <= rules["Good"]:
                    rank = "Good"
                elif val_abs <= rules["Average"]:
                    rank = "Average"
                else:
                    rank = "Poor"

        scorecard_rows.append({
            "Metric": metric_name.replace("_", " "),
            "Value": f"{val:.2f}%" if "Rate" in metric_name or "Drawdown" in metric_name else f"{val:.4f}" if val is not None else "N/A",
            "Grade": rank
        })

    return pd.DataFrame(scorecard_rows)

def plot_performance_dashboard(
    portfolio_df: pd.DataFrame,
    trade_book: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    output_dir: str,
    prefix: str
) -> None:
    """
    Generates 20 publication-quality charts for portfolio, returns, risk, and trade book analysis.

    Args:
        portfolio_df (pd.DataFrame): Strategy portfolio daily balance.
        trade_book (pd.DataFrame): Trade book log.
        benchmark_df (pd.DataFrame): Benchmark price index.
        output_dir (str): output figures folder.
        prefix (str): File name prefix.
    """
    logger.info(f"Generating 20 performance visualizations for '{prefix}' in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300
    })

    pfx = f"{prefix}_" if prefix else ""
    dates = portfolio_df.index
    returns = portfolio_df["Daily_Return"]
    equity = portfolio_df["Portfolio_Value"]

    # Align benchmark
    aligned_bench = benchmark_df.reindex(dates).ffill()
    bench_returns = aligned_bench["Close"].pct_change().fillna(0.0)
    bench_cum = (1.0 + bench_returns).cumprod()

    # 1. Portfolio Equity Curve
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(dates, equity, color="#1f77b4", linewidth=1.5)
    ax.set_title("Portfolio Equity Curve", fontweight="bold")
    ax.set_ylabel("Rupees (INR)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_portfolio_value.png"), dpi=300)
    plt.close(fig)

    # 2. Strategy vs Buy & Hold (Cumulative Returns)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    strat_cum = (1.0 + returns).cumprod() - 1.0
    bench_cum_ret = bench_cum - 1.0
    ax.plot(dates, strat_cum * 100.0, label="Strategy Returns", color="#1f77b4", linewidth=1.5)
    ax.plot(dates, bench_cum_ret * 100.0, label="Nifty 50 Buy & Hold", color="#7f7f7f", linestyle="--", linewidth=1.2)
    ax.set_title("Strategy vs. Buy & Hold Benchmark", fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_vs_benchmark.png"), dpi=300)
    plt.close(fig)

    # 3. Drawdown Underwater Plot
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    drawdowns = calculate_drawdown_series(equity)
    ax.fill_between(dates, -drawdowns * 100.0, 0.0, color="#d62728", alpha=0.3)
    ax.plot(dates, -drawdowns * 100.0, color="#d62728", linewidth=0.8)
    ax.set_title("Underwater Drawdown Plot", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_drawdown.png"), dpi=300)
    plt.close(fig)

    # 4. Daily Return Distribution
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.hist(returns * 100.0, bins=50, density=True, color="#bcbd22", edgecolor="black", alpha=0.6)
    # Overlay normal distribution fit
    mu, std = returns.mean() * 100.0, returns.std() * 100.0
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, "k", linewidth=1.5, label="Normal Fit")
    ax.set_title("Daily Return Distribution", fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_daily_returns_distribution.png"), dpi=300)
    plt.close(fig)

    # 5. Monthly Return Distribution Boxplot
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    # Resample daily returns to monthly
    monthly_ret = returns.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    monthly_df = pd.DataFrame({"Return": monthly_ret * 100.0, "Month": monthly_ret.index.month})
    monthly_data = [monthly_df[monthly_df["Month"] == m]["Return"].values for m in range(1, 13)]
    ax.boxplot(monthly_data, tick_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Monthly Return Distribution (Boxplots by Month)", fontweight="bold")
    ax.set_ylabel("Monthly Return (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_monthly_returns_boxplot.png"), dpi=300)
    plt.close(fig)

    # Rolling metrics computation for charts 6, 7, 8
    rolling_df = calculate_rolling_metrics(portfolio_df, window=126)

    # 6. Rolling Sharpe Ratio
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(rolling_df.index, rolling_df["Rolling_Sharpe"], color="#9467bd", linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Rolling Sharpe Ratio (126-Day Moving Window)", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_rolling_sharpe.png"), dpi=300)
    plt.close(fig)

    # 7. Rolling Volatility
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(rolling_df.index, rolling_df["Rolling_Volatility"] * 100.0, color="#e377c2", linewidth=1.2)
    ax.set_title("Rolling Volatility (126-Day Moving Window)", fontweight="bold")
    ax.set_ylabel("Volatility (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_rolling_volatility.png"), dpi=300)
    plt.close(fig)

    # 8. Rolling CAGR
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(rolling_df.index, rolling_df["Rolling_CAGR"] * 100.0, color="#17becf", linewidth=1.2)
    ax.set_title("Rolling CAGR (126-Day Moving Window)", fontweight="bold")
    ax.set_ylabel("CAGR (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_rolling_cagr.png"), dpi=300)
    plt.close(fig)

    # 9. Return Histogram (Simple hist)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.hist(returns * 100.0, bins=40, color="#2ca02c", edgecolor="black")
    ax.set_title("Daily Return Histogram", fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_return_histogram.png"), dpi=300)
    plt.close(fig)

    # 10. QQ Plot
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    stats.probplot(returns, dist="norm", plot=ax)
    ax.get_lines()[0].set_color("#1f77b4")
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[1].set_color("red")
    ax.set_title("Quantile-Quantile (Q-Q) Plot", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_qq_plot.png"), dpi=300)
    plt.close(fig)

    # 11. Cumulative Return Timeline
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(dates, strat_cum * 100.0, color="#2ca02c", linewidth=1.5)
    ax.set_title("Cumulative Strategy Return", fontweight="bold")
    ax.set_ylabel("Return (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_cumulative_return.png"), dpi=300)
    plt.close(fig)

    # 12. Annual Return Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    annual_ret = returns.groupby(returns.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in annual_ret]
    bars = ax.bar(annual_ret.index, annual_ret * 100.0, color=colors, edgecolor="black")
    ax.set_title("Annual Calendar Returns", fontweight="bold")
    ax.set_ylabel("Return (%)")
    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + (1 if h >= 0 else -3), f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_annual_returns.png"), dpi=300)
    plt.close(fig)

    # 13. Monthly Return Heatmap
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    monthly_ret_matrix = returns.groupby([returns.index.year, returns.index.month]).apply(lambda x: (1.0 + x).prod() - 1.0).unstack()
    monthly_ret_matrix = monthly_ret_matrix * 100.0
    monthly_ret_matrix.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    im = ax.imshow(monthly_ret_matrix, cmap="RdYlGn", aspect="auto", vmin=-10.0, vmax=10.0)
    ax.set_xticks(range(12))
    ax.set_xticklabels(monthly_ret_matrix.columns)
    ax.set_yticks(range(len(monthly_ret_matrix)))
    ax.set_yticklabels(monthly_ret_matrix.index)
    # Add numbers in cells
    for y in range(len(monthly_ret_matrix)):
        for x in range(12):
            val = monthly_ret_matrix.iloc[y, x]
            if not pd.isna(val):
                ax.text(x, y, f"{val:.1f}%", ha="center", va="center", color="black", fontsize=8)
    ax.set_title("Monthly Return Heatmap (%)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Return (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_monthly_heatmap.png"), dpi=300)
    plt.close(fig)

    # 14. Trade Return Distribution
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    if not trade_book.empty:
        trade_returns = trade_book["Position Size"] * (trade_book["Exit Price"] - trade_book["Entry Price"]) / trade_book["Entry Price"]
        cost_returns = trade_book["Total Cost"] / (100000.0 * trade_book["Position Size"].abs())
        net_trade_returns = (trade_returns - cost_returns) * 100.0
        ax.hist(net_trade_returns, bins=20, color="#bcbd22", edgecolor="black")
        ax.axvline(net_trade_returns.mean(), color="red", linestyle="--", label=f"Mean: {net_trade_returns.mean():.2f}%")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No Trades to Plot", ha="center", va="center")
    ax.set_title("Trade Return Distribution", fontweight="bold")
    ax.set_xlabel("Net Trade Return (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_trade_returns_distribution.png"), dpi=300)
    plt.close(fig)

    # 15. Drawdown Duration Plot
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    dd_events = analyze_drawdowns(equity)
    if not dd_events.empty:
        ax.scatter(dd_events["Depth (%)"], dd_events["Duration (TD)"], color="#d62728", s=50, alpha=0.7)
        ax.set_xlabel("Drawdown Depth (%)")
        ax.set_ylabel("Duration (Trading Days)")
        # Annotate top 3 worst
        top_3 = dd_events.head(3)
        for _, row in top_3.iterrows():
            ax.annotate(row["Start Date"].strftime("%Y-%m"), (row["Depth (%)"], row["Duration (TD)"]), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No Drawdown Events", ha="center", va="center")
    ax.set_title("Drawdown Event Depth vs. Duration", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_drawdown_duration.png"), dpi=300)
    plt.close(fig)

    # 16. Risk vs Return Scatter
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    # Annualized return and volatility
    strat_cagr = calculate_return_metrics(portfolio_df["Portfolio_Value"], returns).get("CAGR", 0.0) * 100.0
    strat_vol = calculate_volatility(returns) * 100.0
    bench_cagr = calculate_return_metrics(aligned_bench["Close"], bench_returns).get("CAGR", 0.0) * 100.0
    bench_vol = calculate_volatility(bench_returns) * 100.0

    ax.scatter(strat_vol, strat_cagr, color="#1f77b4", s=100, label="Strategy")
    ax.scatter(bench_vol, bench_cagr, color="#7f7f7f", s=100, marker="X", label="Benchmark (Nifty 50)")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_xlim(min(strat_vol, bench_vol) - 5.0, max(strat_vol, bench_vol) + 5.0)
    ax.set_ylim(min(strat_cagr, bench_cagr) - 5.0, max(strat_cagr, bench_cagr) + 5.0)
    ax.legend()
    ax.set_title("Risk vs. Return Space", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_risk_return_scatter.png"), dpi=300)
    plt.close(fig)

    # 17. Benchmark Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    labels = ["CAGR (%)", "Volatility (%)", "Sharpe", "Max DD (%)"]
    strat_vals = [
        strat_cagr,
        strat_vol,
        calculate_risk_adjusted_ratios(returns, equity).get("Sharpe_Ratio", 0.0),
        calculate_maximum_drawdown(equity)[0] * 100.0
    ]
    bench_vals = [
        bench_cagr,
        bench_vol,
        calculate_risk_adjusted_ratios(bench_returns, aligned_bench["Close"]).get("Sharpe_Ratio", 0.0),
        calculate_maximum_drawdown(aligned_bench["Close"])[0] * 100.0
    ]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, strat_vals, width, label="Strategy", color="#1f77b4")
    ax.bar(x + width/2, bench_vals, width, label="Nifty 50", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Strategy vs. Benchmark Metric Comparison", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_benchmark_comparison.png"), dpi=300)
    plt.close(fig)

    # 18. Rolling Correlation
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    rolling_corr = returns.rolling(126).corr(bench_returns)
    ax.plot(rolling_corr.index, rolling_corr, color="#d62728", linewidth=1.2)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_title("Rolling Correlation with Benchmark (126-Day Window)", fontweight="bold")
    ax.set_ylabel("Pearson Correlation")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_rolling_correlation.png"), dpi=300)
    plt.close(fig)

    # 19. Return Boxplot (Returns by day of week)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    returns_by_day = [returns[returns.index.dayofweek == d].values * 100.0 for d in range(5)]
    ax.boxplot(returns_by_day, tick_labels=["Mon", "Tue", "Wed", "Thu", "Fri"])
    ax.set_title("Returns Distribution by Day of Week", fontweight="bold")
    ax.set_ylabel("Daily Return (%)")
    fig.savefig(os.path.join(output_dir, f"{pfx}perf_returns_boxplot.png"), dpi=300)
    plt.close(fig)

    # 20. Performance Dashboard (A unified 2x3 dashboard summarizing everything in a single figure)
    fig = plt.figure(figsize=(15, 10))
    # Grid: 2 rows, 3 columns
    # Panel 1: Equity Curve vs Benchmark
    ax_dash1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_dash1.plot(dates, strat_cum * 100.0, label="Strategy Returns", color="#1f77b4")
    ax_dash1.plot(dates, bench_cum_ret * 100.0, label="Nifty 50 Buy & Hold", color="#7f7f7f", linestyle="--")
    ax_dash1.set_title("Cumulative Growth vs Benchmark (%)", fontweight="bold")
    ax_dash1.legend()
    
    # Panel 2: Underwater Drawdown
    ax_dash2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax_dash2.fill_between(dates, -drawdowns * 100.0, 0.0, color="#d62728", alpha=0.3)
    ax_dash2.plot(dates, -drawdowns * 100.0, color="#d62728", linewidth=0.8)
    ax_dash2.set_title("Drawdown Timeline (%)", fontweight="bold")
    
    # Panel 3: Monthly Returns Heatmap
    ax_dash3 = plt.subplot2grid((3, 2), (2, 0))
    # Keep it simple, just plot annual returns as a bar chart here to avoid matrix crowding in a dashboard panel
    ax_dash3.bar(annual_ret.index, annual_ret * 100.0, color="#2ca02c", edgecolor="black")
    ax_dash3.set_title("Annual Returns (%)", fontweight="bold")

    # Panel 4: Daily Returns Distribution
    ax_dash4 = plt.subplot2grid((3, 2), (2, 1))
    ax_dash4.hist(returns * 100.0, bins=40, color="#bcbd22", edgecolor="black")
    ax_dash4.set_title("Daily Return Distribution (%)", fontweight="bold")

    fig.suptitle(f"{prefix.upper()} STRATEGY PERFORMANCE DASHBOARD", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{pfx}performance_dashboard.png"), dpi=300)
    plt.close(fig)

    logger.info(f"Successfully generated all 20 charts for '{prefix}'.")
