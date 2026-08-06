"""
Market Regime Visualization Module.

Generates 20 publication-quality charts for market regime timeline,
transition matrices, durations, event overlays, and comparison dashboards.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_regime_analysis(
    prices: pd.DataFrame,
    regimes_df: pd.DataFrame,
    mom_perf: pd.DataFrame,
    mr_perf: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    stability_stats: Dict[str, Any],
    feature_importance: pd.Series,
    output_dir: str
) -> None:
    """
    Generates 20 publication-quality visualizations for market regime analysis.

    Plots generated:
    1. Regime-Colored Price Chart
    2. Momentum Performance by Regime
    3. Mean Reversion Performance by Regime
    4. Regime Timeline
    5. Regime Duration Histogram
    6. Volatility Regimes
    7. Trend Strength Chart
    8. Strategy Return by Regime
    9. Rolling Regime Classification
    10. Transition Matrix Heatmap
    11. Regime Frequency
    12. Regime Calendar
    13. Trade Density by Regime
    14. Exposure by Regime
    15. Performance Radar Chart
    16. Regime Performance Dashboard
    17. Rolling Regime Probability
    18. Market State Timeline
    19. Historical Event Overlay
    20. Comprehensive Regime Dashboard

    Args:
        prices (pd.DataFrame): Nifty 50 price history.
        regimes_df (pd.DataFrame): Combined and final regime labels.
        mom_perf (pd.DataFrame): Momentum performance per regime.
        mr_perf (pd.DataFrame): Mean Reversion performance per regime.
        transition_matrix (pd.DataFrame): Markov transitions table.
        stability_stats (Dict[str, Any]): Durations and stability stats.
        feature_importance (pd.Series): Feature correlations.
        output_dir (str): output directory.
    """
    logger.info("Plotting 20 regime-colored, transition, and dashboard charts...")
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300
    })

    dates = prices.index
    close = prices["Close"]
    regime = regimes_df["Final_Regime"]
    vol_regime = regimes_df["Volatility_Regime"]

    # Color scheme for regimes
    color_map = {
        "Trending_Up_Low_Vol": "#2ca02c",      # Green
        "Trending_Up_Medium_Vol": "#55a868",   # Muted Green
        "Trending_Up_High_Vol": "#85c1e9",     # Light Blue
        "Trending_Down_Low_Vol": "#d98880",    # Light Red
        "Trending_Down_Medium_Vol": "#ec7063", # Muted Red
        "Trending_Down_High_Vol": "#c0392b",   # Red
        "Sideways_Low_Vol": "#d5f5e3",         # Pastel Green
        "Sideways_Medium_Vol": "#fcf3cf",      # Yellow
        "Sideways_High_Vol": "#f39c12",        # Orange
        "Crash": "#78281f",                    # Dark Red/Brown
        "Recovery": "#af7ac5"                  # Purple
    }

    # 1. Regime-Colored Price Chart
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(dates, close, color="black", linewidth=1.0, label="Nifty 50")
    # Shading backgrounds
    for label, color in color_map.items():
        mask = regime == label
        if mask.any():
            # Fill between requires contiguous segments. For simple visualization:
            ax.fill_between(dates, close.min(), close.max(), where=mask, color=color, alpha=0.3, label=label.replace("_", " "))
    ax.set_title("Nifty 50 Index Color-Shaded by Combined Market Regime", fontweight="bold")
    ax.set_ylabel("Price")
    # Simplify legend by removing duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.savefig(os.path.join(output_dir, "reg_colored_price.png"), dpi=300)
    plt.close(fig)

    # 2. Momentum Performance by Regime
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x_mom = np.arange(len(mom_perf))
    ax.bar(x_mom, mom_perf["Sharpe Ratio"], color="#1f77b4", edgecolor="black")
    ax.set_xticks(x_mom)
    ax.set_xticklabels(mom_perf["Regime"], rotation=45, ha="right")
    ax.set_title("Momentum Strategy Sharpe Ratio by Market Regime", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    fig.savefig(os.path.join(output_dir, "reg_momentum_performance.png"), dpi=300)
    plt.close(fig)

    # 3. Mean Reversion Performance by Regime
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x_mr = np.arange(len(mr_perf))
    ax.bar(x_mr, mr_perf["Sharpe Ratio"], color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x_mr)
    ax.set_xticklabels(mr_perf["Regime"], rotation=45, ha="right")
    ax.set_title("Mean Reversion Strategy Sharpe Ratio by Market Regime", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    fig.savefig(os.path.join(output_dir, "reg_mean_reversion_performance.png"), dpi=300)
    plt.close(fig)

    # 4. Regime Timeline (visualizing spans as colored horizontal lines)
    fig, ax = plt.subplots(figsize=(12, 3), layout="constrained")
    # Plot as a flat colored timeline bar
    for label, color in color_map.items():
        mask = regime == label
        if mask.any():
            ax.barh([0], [1], left=dates[mask], height=0.5, color=color, alpha=0.7)
    ax.set_yticks([])
    ax.set_title("Historical Market Regime Timeline", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_timeline.png"), dpi=300)
    plt.close(fig)

    # 5. Regime Duration Histogram
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    # Isolate durations
    blocks = (regime != regime.shift()).cumsum()
    block_lengths = regime.groupby(blocks).size()
    ax.hist(block_lengths, bins=30, color="#bcbd22", edgecolor="black", rwidth=0.8)
    ax.set_xlabel("Duration (Trading Days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Contiguous Regime Spans", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_duration_histogram.png"), dpi=300)
    plt.close(fig)

    # 6. Volatility Regimes
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(dates, close, color="black", linewidth=1.0)
    vol_colors = {"Low_Vol": "green", "Medium_Vol": "yellow", "High_Vol": "red"}
    for label, color in vol_colors.items():
        mask = vol_regime == label
        if mask.any():
            ax.fill_between(dates, close.min(), close.max(), where=mask, color=color, alpha=0.15, label=label)
    ax.set_title("Nifty 50 Volatility States Shading", fontweight="bold")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "reg_volatility_regimes.png"), dpi=300)
    plt.close(fig)

    # 7. Trend Strength Chart
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    trend_strength = (sma_50 - sma_200) / sma_200 * 100.0
    ax.plot(dates, trend_strength, color="#9467bd", linewidth=1.2)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("Trend Strength Indicator (%)")
    ax.set_title("Long-Term Trend Strength Index (50d SMA vs. 200d SMA)", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_trend_strength.png"), dpi=300)
    plt.close(fig)

    # 8. Strategy Return by Regime
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    # Merge on regime
    merged_ret = pd.merge(mom_perf[["Regime", "CAGR (%)"]], mr_perf[["Regime", "CAGR (%)"]], on="Regime", suffixes=("_mom", "_mr"))
    x = np.arange(len(merged_ret))
    width = 0.35
    ax.bar(x - width/2, merged_ret["CAGR (%)_mom"], width, label="Momentum CAGR", color="#1f77b4")
    ax.bar(x + width/2, merged_ret["CAGR (%)_mr"], width, label="Mean Reversion CAGR", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(merged_ret["Regime"], rotation=45, ha="right")
    ax.set_ylabel("Annualized CAGR (%)")
    ax.legend()
    ax.set_title("Strategy Performance comparison by Regime", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_strategy_returns.png"), dpi=300)
    plt.close(fig)

    # 9. Rolling Regime Classification (Rolling 252-day regime percentage weights)
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    # For each day, count frequency in last 252 days
    rolling_counts = pd.get_dummies(regime).rolling(252).mean() * 100.0
    # Plot as stacked area
    # Limit to top 5 most frequent for clarity
    frequent_cols = rolling_counts.mean().sort_values(ascending=False).index[:5]
    ax.stackplot(rolling_counts.index, [rolling_counts[c].fillna(0) for c in frequent_cols], labels=frequent_cols, alpha=0.7)
    ax.set_ylabel("Regime Weight (%)")
    ax.legend(loc="upper left")
    ax.set_title("Rolling 252-Day Regime Composition Breakdown", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_rolling_classification.png"), dpi=300)
    plt.close(fig)

    # 10. Transition Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    if not transition_matrix.empty:
        im = ax.imshow(transition_matrix.values, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(transition_matrix.columns)))
        ax.set_xticklabels(transition_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(transition_matrix.index)))
        ax.set_yticklabels(transition_matrix.index)
        # Numbers inside
        for y in range(len(transition_matrix)):
            for x in range(len(transition_matrix.columns)):
                val = transition_matrix.iloc[y, x]
                if not pd.isna(val):
                    ax.text(x, y, f"{val*100:.1f}%", ha="center", va="center", color="black" if val < 0.5 else "white")
        fig.colorbar(im, ax=ax, label="Transition Probability")
    ax.set_title("Markov State Transition Probability Matrix Heatmap", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_transition_matrix.png"), dpi=300)
    plt.close(fig)

    # 11. Regime Frequency
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    counts = regime.value_counts()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=[color_map.get(k, "#7f7f7f") for k in counts.index])
    ax.set_title("Historical Regime Occurrence Frequency", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_frequency.png"), dpi=300)
    plt.close(fig)

    # 12. Regime Calendar Heatmap (dominant regime for each month/year)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    # For each month and year, find the mode regime label
    mode_regime = regime.groupby([regime.index.year, regime.index.month]).apply(lambda x: x.mode().iloc[0]).unstack()
    # Map text to values for coloring
    all_unique = regime.dropna().unique().tolist()
    regime_to_int = {name: idx for idx, name in enumerate(all_unique)}
    mapped_matrix = mode_regime.replace(regime_to_int).fillna(-1).astype(float)
    
    im = ax.imshow(mapped_matrix.values, cmap="Accent", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(mode_regime)))
    ax.set_yticklabels(mode_regime.index)
    # Add text on cells
    for y in range(len(mode_regime)):
        for x in range(12):
            val_name = mode_regime.iloc[y, x]
            if not pd.isna(val_name):
                # Shorten name
                short_name = val_name.replace("Trending_Up_", "U_").replace("Trending_Down_", "D_").replace("Sideways_", "S_").replace("_Vol", "")
                ax.text(x, y, short_name, ha="center", va="center", fontsize=7)
    ax.set_title("Regime Calendar: Dominant Monthly Market Regimes", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_calendar.png"), dpi=300)
    plt.close(fig)

    # 13. Trade Density by Regime
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    # Slices density: Trade Count / Active Days
    density_mom = mom_perf["Trade Count"] / mom_perf["Active Days"] * 100.0
    density_mr = mr_perf["Trade Count"] / mr_perf["Active Days"] * 100.0
    
    categories = list(mom_perf["Regime"] + " (Mom)") + list(mr_perf["Regime"] + " (MR)")
    heights = list(density_mom) + list(density_mr)
    colors = ["#1f77b4"] * len(density_mom) + ["#ff7f0e"] * len(density_mr)
    
    x_dens = np.arange(len(categories))
    ax.bar(x_dens, heights, color=colors, width=0.6)
    ax.set_xticks(x_dens)
    ax.set_xticklabels(categories, rotation=90, fontsize=8)
    ax.set_ylabel("Trade Density (Trades per 100 Days)")
    ax.set_title("Strategy Trading Frequency Density by Regime", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_trade_density.png"), dpi=300)
    plt.close(fig)

    # 14. Exposure by Regime
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x = np.arange(len(mom_perf))
    ax.bar(x - 0.2, mom_perf["Exposure (%)"], width=0.4, label="Momentum Exposure", color="#1f77b4")
    ax.bar(x + 0.2, mr_perf["Exposure (%)"], width=0.4, label="Mean Reversion Exposure", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(mom_perf["Regime"], rotation=45, ha="right")
    ax.set_ylabel("Portfolio Capital Exposure (%)")
    ax.legend()
    ax.set_title("Strategy Time-in-Market Exposure by Regime", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_exposure.png"), dpi=300)
    plt.close(fig)

    # 15. Performance Radar Chart across Regimes (CAGR comparison)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    regimes_list = merged_ret["Regime"].tolist()
    angles = np.linspace(0, 2*np.pi, len(regimes_list), endpoint=False).tolist()
    angles += angles[:1]
    mom_radar = merged_ret["CAGR (%)_mom"].tolist()
    mom_radar += mom_radar[:1]
    mr_radar = merged_ret["CAGR (%)_mr"].tolist()
    mr_radar += mr_radar[:1]
    ax.plot(angles, mom_radar, color="#1f77b4", linewidth=2, label="Momentum CAGR")
    ax.fill(angles, mom_radar, color="#1f77b4", alpha=0.2)
    ax.plot(angles, mr_radar, color="#ff7f0e", linewidth=2, label="Mean Reversion CAGR")
    ax.fill(angles, mr_radar, color="#ff7f0e", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(regimes_list, fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    ax.set_title("Annualized CAGR Comparison across Regimes", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_radar_chart.png"), dpi=300)
    plt.close(fig)

    # 16. Regime Performance Dashboard (Multi-panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_mom = np.arange(len(mom_perf))
    x_mr = np.arange(len(mr_perf))
    
    # Sharpe Ratio Comparison
    axes[0, 0].bar(x_mom, mom_perf["Sharpe Ratio"], color="#1f77b4", alpha=0.8, label="Momentum")
    axes[0, 0].set_xticks(x_mom)
    axes[0, 0].set_xticklabels(mom_perf["Regime"], rotation=30, ha="right", fontsize=8)
    axes[0, 0].set_title("Sharpe Ratio: Momentum", fontweight="bold")
    
    axes[0, 1].bar(x_mr, mr_perf["Sharpe Ratio"], color="#ff7f0e", alpha=0.8, label="Mean Reversion")
    axes[0, 1].set_xticks(x_mr)
    axes[0, 1].set_xticklabels(mr_perf["Regime"], rotation=30, ha="right", fontsize=8)
    axes[0, 1].set_title("Sharpe Ratio: Mean Reversion", fontweight="bold")

    # Max DD Comparison
    axes[1, 0].bar(x_mom, mom_perf["Max Drawdown (%)"], color="#c0392b", alpha=0.7)
    axes[1, 0].set_xticks(x_mom)
    axes[1, 0].set_xticklabels(mom_perf["Regime"], rotation=30, ha="right", fontsize=8)
    axes[1, 0].set_title("Max Drawdown: Momentum (%)", fontweight="bold")

    axes[1, 1].bar(x_mr, mr_perf["Max Drawdown (%)"], color="#78281f", alpha=0.7)
    axes[1, 1].set_xticks(x_mr)
    axes[1, 1].set_xticklabels(mr_perf["Regime"], rotation=30, ha="right", fontsize=8)
    axes[1, 1].set_title("Max Drawdown: Mean Reversion (%)", fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reg_performance_dashboard.png"), dpi=300)
    plt.close(fig)

    # 17. Rolling Regime Probability
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    # Rolling 126-day probability of being in Trending Up or High Vol
    trend_up_prob = (regime.str.contains("Trending_Up")).rolling(126).mean() * 100.0
    high_vol_prob = (regime.str.contains("High_Vol") | (regime == "Crash")).rolling(126).mean() * 100.0
    ax.plot(dates, trend_up_prob, label="Rolling Trending Up Probability", color="#2ca02c")
    ax.plot(dates, high_vol_prob, label="Rolling High Volatility/Crash Probability", color="#d62728")
    ax.set_ylabel("Probability (%)")
    ax.legend()
    ax.set_title("Rolling 126-Day Regime State Probabilities", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_rolling_probability.png"), dpi=300)
    plt.close(fig)

    # 18. Market State Timeline (Simple line showing state index over time)
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    # map states to numbers
    state_names = list(color_map.keys())
    state_mapping = {n: i for i, n in enumerate(state_names)}
    state_timeline = regime.map(state_mapping).ffill()
    ax.step(dates, state_timeline, color="navy", where="pre", linewidth=0.8)
    ax.set_yticks(range(len(state_names)))
    ax.set_yticklabels(state_names, fontsize=7)
    ax.set_title("Combined Market State Sequence Timeline", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_market_state_timeline.png"), dpi=300)
    plt.close(fig)

    # 19. Historical Event Overlay
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(dates, close, color="black", linewidth=1.0)
    # Highlight events
    events = [
        ("COVID-19 Crash", "2020-03-23", 7610.0, "down"),
        ("2016 Volatility (Brexit/Demonetisation)", "2016-11-09", 8400.0, "up"),
        ("2018 NBFC Crisis Correction", "2018-10-26", 10030.0, "down"),
        ("2022 Global Rate Hike Decline", "2022-06-17", 15290.0, "down")
    ]
    for label, date_str, price, direction in events:
        evt_date = pd.Timestamp(date_str)
        if evt_date in dates:
            xytext = (0, -30) if direction == "down" else (0, 30)
            ax.annotate(label, xy=(evt_date, price), xytext=xytext, textcoords="offset points", arrowprops=dict(facecolor="red", arrowstyle="->"), ha="center", fontsize=8, fontweight="bold")
    ax.set_title("Historical Market Crises and Corrections Annotated", fontweight="bold")
    fig.savefig(os.path.join(output_dir, "reg_historical_event_overlay.png"), dpi=300)
    plt.close(fig)

    # 20. Comprehensive Regime Dashboard (Unified)
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Timeline
    ax_dash1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_dash1.plot(dates, close, color="black", linewidth=0.8)
    for label, color in color_map.items():
        mask = regime == label
        if mask.any():
            ax_dash1.fill_between(dates, close.min(), close.max(), where=mask, color=color, alpha=0.2)
    ax_dash1.set_title("Nifty 50 Shaded by Combined Market Regime", fontweight="bold")
    
    # 2. Performance Comparison (CAGR)
    ax_dash2 = plt.subplot2grid((3, 2), (1, 0))
    ax_dash2.bar(x - width/2, merged_ret["CAGR (%)_mom"], width, label="Momentum", color="#1f77b4")
    ax_dash2.bar(x + width/2, merged_ret["CAGR (%)_mr"], width, label="Mean Reversion", color="#ff7f0e")
    ax_dash2.set_xticks(x)
    ax_dash2.set_xticklabels(merged_ret["Regime"], rotation=30, ha="right", fontsize=7)
    ax_dash2.set_ylabel("CAGR (%)")
    ax_dash2.legend(fontsize=8)
    ax_dash2.set_title("CAGR comparison by Regime", fontweight="bold")

    # 3. Transition Matrix Heatmap
    ax_dash3 = plt.subplot2grid((3, 2), (1, 1))
    if not transition_matrix.empty:
        im = ax_dash3.imshow(transition_matrix.values, cmap="Blues", aspect="auto")
        ax_dash3.set_xticks(range(len(transition_matrix.columns)))
        ax_dash3.set_xticklabels(transition_matrix.columns, rotation=30, ha="right", fontsize=7)
        ax_dash3.set_yticks(range(len(transition_matrix.index)))
        ax_dash3.set_yticklabels(transition_matrix.index, fontsize=7)
    ax_dash3.set_title("Markov State Transition Matrix", fontweight="bold")

    # 4. Feature correlation
    ax_dash4 = plt.subplot2grid((3, 2), (2, 0))
    ax_dash4.barh(feature_importance.index, feature_importance.values, color="#2ca02c")
    ax_dash4.set_xlabel("Absolute Correlation Coefficient")
    ax_dash4.set_title("Feature Importance: Market Drivers vs. Strategy Returns", fontweight="bold")

    # 5. Stability/Durations
    ax_dash5 = plt.subplot2grid((3, 2), (2, 1))
    ax_dash5.hist(block_lengths, bins=20, color="#bcbd22", edgecolor="black")
    ax_dash5.set_xlabel("Regime Span (Trading Days)")
    ax_dash5.set_title("Regime Persistence Span Distribution", fontweight="bold")

    fig.suptitle("MARKET REGIME ANALYSIS SYSTEM DASHBOARD", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reg_comprehensive_dashboard.png"), dpi=300)
    plt.close(fig)

    logger.info("Successfully generated all 20 market regime analysis visualizations.")
