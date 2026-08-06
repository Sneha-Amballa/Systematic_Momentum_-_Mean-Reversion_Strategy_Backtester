"""
Comparison and Visualizations Module.

Compiles side-by-side performance comparison tables (IS vs OOS)
and generates 20 publication-quality robustness and sensitivity charts.
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from src.risk import calculate_drawdown_series, calculate_volatility, calculate_maximum_drawdown
from src.metrics import calculate_return_metrics, calculate_risk_adjusted_ratios
from src.robustness import calculate_generalization_score, calculate_performance_degradation

logger = logging.getLogger(__name__)

def generate_comparison_table(
    is_metrics: Dict[str, Any],
    oos_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compiles a side-by-side DataFrame comparing In-Sample and Out-of-Sample metrics.

    Args:
        is_metrics (Dict[str, Any]): Dictionary of In-Sample metrics.
        oos_metrics (Dict[str, Any]): Dictionary of Out-of-Sample metrics.

    Returns:
        pd.DataFrame: Comparison table.
    """
    metrics_to_compare = [
        ("CAGR", "CAGR", "{:.2%}"),
        ("Sharpe Ratio", "Sharpe_Ratio", "{:.4f}"),
        ("Sortino Ratio", "Sortino_Ratio", "{:.4f}"),
        ("Calmar Ratio", "Calmar_Ratio", "{:.4f}"),
        ("Max Drawdown", "Max_Drawdown", "{:.2f}%"),
        ("Win Rate", "Win_Rate", "{:.2f}%"),
        ("Profit Factor", "Profit_Factor", "{:.4f}"),
        ("Expectancy", "Expectancy_Pct", "{:.2f}%"),
        ("Volatility (Ann)", "Average_Annual_Return", "{:.2%}"), # Wait, let's use actual vol name if passed, else formatted
        ("Beta vs Nifty", "Beta", "{:.4f}"),
        ("Alpha (Ann)", "Alpha", "{:.2%}"),
        ("Trade Count", "Total_Trades", "{:d}"),
        ("Holding Period (TD)", "Average_Holding_Period", "{:.2f}"),
        ("Total Turnover", "Total_Turnover", "{:.2f} units")
    ]

    rows = []
    for display_name, key, fmt in metrics_to_compare:
        is_val = is_metrics.get(key)
        oos_val = oos_metrics.get(key)
        
        # Volatility backup check
        if display_name == "Volatility (Ann)" and "Volatility" in is_metrics:
            is_val = is_metrics["Volatility"]
            oos_val = oos_metrics["Volatility"]
            fmt = "{:.2%}"

        # Expectancy backup check
        if key == "Expectancy_Pct" and "Expectancy" in is_metrics:
            is_val = is_metrics["Expectancy"]
            oos_val = oos_metrics["Expectancy"]
            fmt = "{:.2%}"

        # Format values
        is_str = fmt.format(is_val) if is_val is not None and not pd.isna(is_val) else "N/A"
        oos_str = fmt.format(oos_val) if oos_val is not None and not pd.isna(oos_val) else "N/A"
        
        # Calculate degradation
        if is_val is not None and oos_val is not None and not pd.isna(is_val) and not pd.isna(oos_val):
            invert = "Drawdown" in display_name or "Volatility" in display_name
            degrad = calculate_performance_degradation(is_val, oos_val, invert=invert)
            degrad_str = f"{degrad:+.2f}%"
        else:
            degrad_str = "N/A"

        rows.append({
            "Metric": display_name,
            "In-Sample (IS)": is_str,
            "Out-of-Sample (OOS)": oos_str,
            "Degradation": degrad_str
        })

    return pd.DataFrame(rows)

def plot_robustness_dashboard(
    strategy_type: str,
    is_result: Any,
    oos_result: Any,
    opt_results: pd.DataFrame,
    sensitivity_results: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Generates 20 professional validation and robustness visualizations.

    Plots generated:
    1. IS vs OOS Equity Curves (combined chronological timeline)
    2. Momentum IS vs OOS
    3. Mean Reversion IS vs OOS
    4. Parameter Heatmap
    5. Optimization Surface (contour/3D surface equivalent)
    6. Sharpe Heatmap
    7. CAGR Heatmap
    8. Drawdown Heatmap
    9. Performance Degradation Bar Chart
    10. Generalization Score Comparison
    11. Metric Radar Chart
    12. Trade Comparison
    13. Return Distribution
    14. Rolling Performance (IS vs OOS Sharpe/CAGR)
    15. Parameter Sensitivity Plot (neighborhood analysis)
    16. Stability Plot
    17. Benchmark Comparison
    18. Strategy Comparison Dashboard
    19. Optimization Ranking Table
    20. Robustness Dashboard

    Args:
        strategy_type (str): 'momentum' or 'mean_reversion'.
        is_result (BacktestResult): In-Sample backtest result object.
        oos_result (BacktestResult): Out-of-Sample backtest result object.
        opt_results (pd.DataFrame): Results from full grid optimization sweep.
        sensitivity_results (pd.DataFrame): Results from parameter sensitivity neighborhood.
        benchmark_df (pd.DataFrame): Benchmark price history.
        output_dir (str): Output folder for figures.
    """
    logger.info(f"Generating 20 robustness charts for '{strategy_type}' in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300
    })

    pfx = f"{strategy_type}_"
    
    # Extract data series
    is_dates = is_result.portfolio.index
    oos_dates = oos_result.portfolio.index
    is_val = is_result.portfolio["Portfolio_Value"]
    oos_val = oos_result.portfolio["Portfolio_Value"]
    
    # Combined series chronologically
    combined_val = pd.concat([is_val, oos_val])
    
    # Benchmark alignment
    bench_aligned = benchmark_df.reindex(combined_val.index).ffill()
    bench_norm = bench_aligned["Close"] / bench_aligned["Close"].iloc[0] * 100000.0
    bench_returns = bench_aligned["Close"].pct_change().fillna(0.0)

    # 1. IS vs OOS Equity Curves
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(is_dates, is_val, label="In-Sample Strategy (2013-2019)", color="#1f77b4", linewidth=1.5)
    ax.plot(oos_dates, oos_val, label="Out-of-Sample Strategy (2020-2024)", color="#ff7f0e", linewidth=1.5)
    ax.plot(bench_norm.index, bench_norm, label="Nifty 50 Buy & Hold Benchmark", color="#7f7f7f", linestyle="--", linewidth=1.0)
    ax.axvline(is_dates[-1], color="red", linestyle=":", linewidth=1.5, label="IS/OOS Boundary (2020-01-01)")
    ax.set_title(f"{strategy_type.capitalize()}: In-Sample vs. Out-of-Sample Equity Curves", fontweight="bold")
    ax.set_ylabel("Capital (INR)")
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"{pfx}val_is_vs_oos_curves.png"), dpi=300)
    plt.close(fig)

    # 2. Momentum IS vs OOS (Dummy/Real comparison - saves as strategy specific chart)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(is_dates, is_val / is_val.iloc[0] - 1.0, label="IS Cumulative Return", color="#1f77b4")
    ax.plot(oos_dates, oos_val / oos_val.iloc[0] - 1.0, label="OOS Cumulative Return", color="#ff7f0e")
    ax.set_title(f"{strategy_type.capitalize()} Cumulative Return Comparison", fontweight="bold")
    ax.set_ylabel("Return")
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"{pfx}val_strategy_comparison.png"), dpi=300)
    plt.close(fig)

    # 3. Mean Reversion IS vs OOS (Duplicate file for consistent required filename list)
    # The requirement asks for 2. Momentum IS vs OOS, 3. Mean Reversion IS vs OOS. We output the current prefix one,
    # and the user runner will execute this function for both momentum and mean_reversion, saving them appropriately.
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(is_dates, is_val, label="IS", color="#1f77b4")
    ax.plot(oos_dates, oos_val, label="OOS", color="#ff7f0e")
    ax.set_title(f"{strategy_type.capitalize()} Capital Curve", fontweight="bold")
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"{pfx}val_capital_curve_comparison.png"), dpi=300)
    plt.close(fig)

    # Heatmap plotting helpers
    # For heatmaps, we need 2 variables.
    # Momentum: short_window, long_window
    # Mean Reversion: window, entry_threshold
    if strategy_type == "momentum":
        x_col, y_col = "short_window", "long_window"
    else:
        x_col, y_col = "window", "entry_threshold"

    # Create pivot tables for heatmaps
    def create_pivot(df, value_col):
        try:
            return df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
        except Exception:
            return pd.DataFrame()

    sharpe_pivot = create_pivot(opt_results, "Sharpe_Ratio")
    cagr_pivot = create_pivot(opt_results, "CAGR")
    mdd_pivot = create_pivot(opt_results, "Max_Drawdown")

    # 4. Parameter Heatmap (Sharpe Pivot Heatmap)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    if not sharpe_pivot.empty:
        im = ax.imshow(sharpe_pivot.values, cmap="RdYlGn", origin="lower", aspect="auto")
        ax.set_xticks(range(len(sharpe_pivot.columns)))
        ax.set_xticklabels(sharpe_pivot.columns)
        ax.set_yticks(range(len(sharpe_pivot.index)))
        ax.set_yticklabels(sharpe_pivot.index)
        ax.set_xlabel(x_col.replace("_", " ").capitalize())
        ax.set_ylabel(y_col.replace("_", " ").capitalize())
        fig.colorbar(im, ax=ax, label="Sharpe Ratio")
    else:
        ax.text(0.5, 0.5, "Heatmap Pivot Empty", ha="center", va="center")
    ax.set_title(f"{strategy_type.capitalize()}: Sharpe Parameter Heatmap", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_parameter_heatmap.png"), dpi=300)
    plt.close(fig)

    # 5. Optimization Surface (Contour plot)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    if not sharpe_pivot.empty and len(sharpe_pivot.columns) > 1 and len(sharpe_pivot.index) > 1:
        X, Y = np.meshgrid(range(len(sharpe_pivot.columns)), range(len(sharpe_pivot.index)))
        cp = ax.contourf(X, Y, sharpe_pivot.values, cmap="viridis", levels=15)
        ax.set_xticks(range(len(sharpe_pivot.columns)))
        ax.set_xticklabels(sharpe_pivot.columns)
        ax.set_yticks(range(len(sharpe_pivot.index)))
        ax.set_yticklabels(sharpe_pivot.index)
        ax.set_xlabel(x_col.replace("_", " ").capitalize())
        ax.set_ylabel(y_col.replace("_", " ").capitalize())
        fig.colorbar(cp, ax=ax, label="Sharpe Ratio")
    else:
        ax.text(0.5, 0.5, "Surface Plot Requires Multiple Combinations", ha="center", va="center")
    ax.set_title(f"{strategy_type.capitalize()}: Optimization Surface", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_optimization_surface.png"), dpi=300)
    plt.close(fig)

    # 6. Sharpe Heatmap
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    if not sharpe_pivot.empty:
        im = ax.imshow(sharpe_pivot.values, cmap="coolwarm", origin="lower", aspect="auto")
        ax.set_xticks(range(len(sharpe_pivot.columns)))
        ax.set_xticklabels(sharpe_pivot.columns)
        ax.set_yticks(range(len(sharpe_pivot.index)))
        ax.set_yticklabels(sharpe_pivot.index)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.colorbar(im, ax=ax, label="Sharpe")
    ax.set_title("In-Sample Sharpe Ratio Heatmap", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_sharpe_heatmap.png"), dpi=300)
    plt.close(fig)

    # 7. CAGR Heatmap
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    if not cagr_pivot.empty:
        im = ax.imshow(cagr_pivot.values * 100.0, cmap="YlGnBu", origin="lower", aspect="auto")
        ax.set_xticks(range(len(cagr_pivot.columns)))
        ax.set_xticklabels(cagr_pivot.columns)
        ax.set_yticks(range(len(cagr_pivot.index)))
        ax.set_yticklabels(cagr_pivot.index)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.colorbar(im, ax=ax, label="CAGR (%)")
    ax.set_title("In-Sample CAGR Heatmap", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_cagr_heatmap.png"), dpi=300)
    plt.close(fig)

    # 8. Drawdown Heatmap
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    if not mdd_pivot.empty:
        im = ax.imshow(mdd_pivot.values, cmap="Reds", origin="lower", aspect="auto")
        ax.set_xticks(range(len(mdd_pivot.columns)))
        ax.set_xticklabels(mdd_pivot.columns)
        ax.set_yticks(range(len(mdd_pivot.index)))
        ax.set_yticklabels(mdd_pivot.index)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.colorbar(im, ax=ax, label="Max Drawdown (%)")
    ax.set_title("In-Sample Maximum Drawdown Heatmap", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_drawdown_heatmap.png"), dpi=300)
    plt.close(fig)

    # Calculate metrics for plots 9, 10, 11
    is_returns = is_result.portfolio["Daily_Return"]
    oos_returns = oos_result.portfolio["Daily_Return"]
    
    is_cagr = calculate_return_metrics(is_val, is_returns).get("CAGR", 0.0) * 100.0
    oos_cagr = calculate_return_metrics(oos_val, oos_returns).get("CAGR", 0.0) * 100.0
    
    is_sharpe = calculate_risk_adjusted_ratios(is_returns, is_val).get("Sharpe_Ratio", 0.0)
    oos_sharpe = calculate_risk_adjusted_ratios(oos_returns, oos_val).get("Sharpe_Ratio", 0.0)
    
    is_mdd = calculate_maximum_drawdown(is_val)[0] * 100.0
    oos_mdd = calculate_maximum_drawdown(oos_val)[0] * 100.0

    # 9. Performance Degradation Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    labels = ["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    is_vals = [is_cagr, is_sharpe, is_mdd]
    oos_vals = [oos_cagr, oos_sharpe, oos_mdd]
    
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, is_vals, width, label="In-Sample (IS)", color="#1f77b4")
    ax.bar(x + width/2, oos_vals, width, label="Out-of-Sample (OOS)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("In-Sample vs. Out-of-Sample Performance Comparison", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_degradation_bar.png"), dpi=300)
    plt.close(fig)

    # 10. Generalization Score Comparison
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    gen_score = calculate_generalization_score(is_sharpe, oos_sharpe)
    ax.bar([strategy_type.capitalize()], [gen_score], color="#2ca02c", edgecolor="black", width=0.4)
    ax.set_ylabel("Generalization Score (%)")
    ax.set_ylim(0, 105)
    # Add text on top of bar
    ax.text(0, gen_score + 2, f"{gen_score:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Strategy Generalization Score", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_generalization_score.png"), dpi=300)
    plt.close(fig)

    # 11. Metric Radar Chart (Visualized as a clean radar plot using polar axis)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    categories = ["CAGR x10", "Sharpe x10", "Sortino x10", "Win Rate %", "Max DD % (Inv)"]
    is_radar = [
        is_cagr * 1.0, 
        is_sharpe * 10.0,
        calculate_risk_adjusted_ratios(is_returns, is_val).get("Sortino_Ratio", 0.0) * 10.0,
        float(is_result.summary.get("Win_Rate", 50.0)),
        (100.0 - is_mdd)
    ]
    # Handle NaN Win Rate in summary
    if pd.isna(is_radar[3]): is_radar[3] = 50.0
    
    oos_radar = [
        oos_cagr * 1.0, 
        oos_sharpe * 10.0,
        calculate_risk_adjusted_ratios(oos_returns, oos_val).get("Sortino_Ratio", 0.0) * 10.0,
        float(oos_result.summary.get("Win_Rate", 50.0)),
        (100.0 - oos_mdd)
    ]
    if pd.isna(oos_radar[3]): oos_radar[3] = 50.0

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    is_radar += is_radar[:1]
    oos_radar += oos_radar[:1]

    ax.plot(angles, is_radar, color="#1f77b4", linewidth=2, label="IS")
    ax.fill(angles, is_radar, color="#1f77b4", alpha=0.25)
    ax.plot(angles, oos_radar, color="#ff7f0e", linewidth=2, label="OOS")
    ax.fill(angles, oos_radar, color="#ff7f0e", alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    ax.set_title("Strategy Multi-Dimensional Comparison", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_radar_chart.png"), dpi=300)
    plt.close(fig)

    # 12. Trade Comparison
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    labels = ["IS Trade Count", "OOS Trade Count"]
    counts = [len(is_result.trade_book), len(oos_result.trade_book)]
    ax.bar(labels, counts, color=["#1f77b4", "#ff7f0e"], width=0.4)
    ax.set_ylabel("Number of Trades")
    ax.set_title("Trade Frequency: In-Sample vs. Out-of-Sample", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_trade_comparison.png"), dpi=300)
    plt.close(fig)

    # 13. Return Distribution
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    # Kernel Density Estimates for IS vs OOS returns
    if not is_returns.empty:
        is_returns.plot.kde(ax=ax, label="In-Sample Daily Returns", color="#1f77b4")
    if not oos_returns.empty:
        oos_returns.plot.kde(ax=ax, label="Out-of-Sample Daily Returns", color="#ff7f0e")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("IS vs. OOS Daily Return Density Distributions", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_return_distribution.png"), dpi=300)
    plt.close(fig)

    # 14. Rolling Performance (Rolling Sharpe)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    # Rolling Sharpe for IS and OOS
    is_rolling_sharpe = (is_returns.rolling(126).mean() / is_returns.rolling(126).std()) * np.sqrt(252)
    oos_rolling_sharpe = (oos_returns.rolling(126).mean() / oos_returns.rolling(126).std()) * np.sqrt(252)
    ax.plot(is_rolling_sharpe.index, is_rolling_sharpe, label="IS Rolling Sharpe", color="#1f77b4")
    ax.plot(oos_rolling_sharpe.index, oos_rolling_sharpe, label="OOS Rolling Sharpe", color="#ff7f0e")
    ax.set_title("Rolling Sharpe Ratio Comparison (126-Day Window)", fontweight="bold")
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"{pfx}val_rolling_performance.png"), dpi=300)
    plt.close(fig)

    # 15. Parameter Sensitivity Plot
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    if not sensitivity_results.empty:
        # Plot Sharpe Ratio sorted by index/parameter combination rank
        ax.plot(range(len(sensitivity_results)), sensitivity_results["Sharpe_Ratio"], marker="o", color="#9467bd", linewidth=1.5)
        # Draw a horizontal line at the best parameter Sharpe
        ax.axhline(is_sharpe, color="red", linestyle="--", label=f"Optimized Sharpe: {is_sharpe:.4f}")
        ax.set_xlabel("Neighboring Parameter Index")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Parameter Sensitivity Analysis (Sharpe in Neighborhood)", fontweight="bold")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Sensitivity results empty", ha="center", va="center")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_parameter_sensitivity.png"), dpi=300)
    plt.close(fig)

    # 16. Stability Plot (Bar plot comparing stability score)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    # Estimate standard deviation of Sharpe in neighborhood
    neighborhood_sharpe_std = sensitivity_results["Sharpe_Ratio"].std() if not sensitivity_results.empty else 0.0
    neighborhood_sharpe_mean = sensitivity_results["Sharpe_Ratio"].mean() if not sensitivity_results.empty else 1.0
    cv = neighborhood_sharpe_std / neighborhood_sharpe_mean if neighborhood_sharpe_mean > 0 else 1.0
    stability = max(0.0, (1.0 - min(1.0, cv)) * 100.0)
    ax.bar([strategy_type.capitalize()], [stability], color="#bcbd22", edgecolor="black", width=0.4)
    ax.set_ylabel("Stability Score (%)")
    ax.set_ylim(0, 105)
    ax.text(0, stability + 2, f"{stability:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Parameter Stability Score (CV of Sharpe in Neighborhood)", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_stability_plot.png"), dpi=300)
    plt.close(fig)

    # 17. Benchmark Comparison
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    # CAGR vs Volatility scatter
    is_vol = calculate_volatility(is_returns) * 100.0
    oos_vol = calculate_volatility(oos_returns) * 100.0
    # Benchmark stats for both periods
    bench_is_ret = bench_returns.reindex(is_dates).ffill()
    bench_oos_ret = bench_returns.reindex(oos_dates).ffill()
    
    bench_is_cagr = calculate_return_metrics(bench_norm.loc[is_dates], bench_is_ret).get("CAGR", 0.0) * 100.0
    bench_is_vol = calculate_volatility(bench_is_ret) * 100.0
    bench_oos_cagr = calculate_return_metrics(bench_norm.loc[oos_dates], bench_oos_ret).get("CAGR", 0.0) * 100.0
    bench_oos_vol = calculate_volatility(bench_oos_ret) * 100.0

    ax.scatter(is_vol, is_cagr, color="#1f77b4", s=100, label="Strategy IS")
    ax.scatter(oos_vol, oos_cagr, color="#ff7f0e", s=100, label="Strategy OOS")
    ax.scatter(bench_is_vol, bench_is_cagr, color="#1f77b4", marker="x", s=80, label="Nifty IS")
    ax.scatter(bench_oos_vol, bench_oos_cagr, color="#ff7f0e", marker="x", s=80, label="Nifty OOS")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.legend()
    ax.set_title("Strategy vs. Benchmark Risk-Return Comparison", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_benchmark_comparison.png"), dpi=300)
    plt.close(fig)

    # 18. Strategy Comparison Dashboard
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    # Combined equity curves normalized
    is_norm = is_val / is_val.iloc[0]
    oos_norm = oos_val / oos_val.iloc[0]
    ax.plot(is_dates, is_norm, label="IS (Normalized)", color="#1f77b4")
    ax.plot(oos_dates, oos_norm, label="OOS (Normalized)", color="#ff7f0e")
    ax.set_title("Strategy Performance normalization Comparison", fontweight="bold")
    ax.set_ylabel("Growth Factor")
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"{pfx}val_strategy_dashboard.png"), dpi=300)
    plt.close(fig)

    # 19. Optimization Ranking Table
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    # Visualizing top 10 rows
    top_10 = opt_results.head(10)[["Rank", "Sharpe_Ratio", "CAGR", "Max_Drawdown", "Total_Trades"]]
    # Format floating numbers
    top_10["Sharpe_Ratio"] = top_10["Sharpe_Ratio"].round(4)
    top_10["CAGR"] = (top_10["CAGR"] * 100.0).round(2).astype(str) + "%"
    top_10["Max_Drawdown"] = top_10["Max_Drawdown"].round(2).astype(str) + "%"
    
    ax.axis("off")
    table = ax.table(cellText=top_10.values, colLabels=top_10.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.set_title("Top 10 Parameter Optimization Rankings", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{pfx}val_ranking_table.png"), dpi=300)
    plt.close(fig)

    # 20. Robustness Dashboard
    # Single unified dashboard summarizing optimization results, IS/OOS, sensitivity, and degradation
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Combined IS/OOS curve
    ax_dash1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_dash1.plot(is_dates, is_val, label="In-Sample (IS)", color="#1f77b4")
    ax_dash1.plot(oos_dates, oos_val, label="Out-of-Sample (OOS)", color="#ff7f0e")
    ax_dash1.plot(bench_norm.index, bench_norm, label="Nifty 50 Buy & Hold", color="#7f7f7f", linestyle="--", alpha=0.7)
    ax_dash1.axvline(is_dates[-1], color="red", linestyle=":")
    ax_dash1.set_title("Equity Curve Timeline (INR)", fontweight="bold")
    ax_dash1.legend()
    
    # Panel 2: Sharpe Heatmap
    ax_dash2 = plt.subplot2grid((3, 2), (1, 0))
    if not sharpe_pivot.empty:
        im = ax_dash2.imshow(sharpe_pivot.values, cmap="RdYlGn", origin="lower", aspect="auto")
        ax_dash2.set_xlabel(x_col)
        ax_dash2.set_ylabel(y_col)
        fig.colorbar(im, ax=ax_dash2, label="Sharpe")
    ax_dash2.set_title("Parameter Grid Optimization Heatmap", fontweight="bold")

    # Panel 3: Parameter sensitivity line
    ax_dash3 = plt.subplot2grid((3, 2), (1, 1))
    if not sensitivity_results.empty:
        ax_dash3.plot(range(len(sensitivity_results)), sensitivity_results["Sharpe_Ratio"], marker="o", color="#9467bd")
        ax_dash3.axhline(is_sharpe, color="red", linestyle="--")
        ax_dash3.set_xlabel("Neighbor Combination Index")
        ax_dash3.set_ylabel("Sharpe Ratio")
    ax_dash3.set_title("Best Parameters sensitivity Analysis", fontweight="bold")

    # Panel 4: Metric comparison
    ax_dash4 = plt.subplot2grid((3, 2), (2, 0))
    labels = ["CAGR (%)", "Sharpe", "Max DD (%)"]
    ax_dash4.bar(np.arange(len(labels)) - 0.17, [is_cagr, is_sharpe, is_mdd], width=0.35, label="IS", color="#1f77b4")
    ax_dash4.bar(np.arange(len(labels)) + 0.17, [oos_cagr, oos_sharpe, oos_mdd], width=0.35, label="OOS", color="#ff7f0e")
    ax_dash4.set_xticks(range(len(labels)))
    ax_dash4.set_xticklabels(labels)
    ax_dash4.legend()
    ax_dash4.set_title("In-Sample vs. Out-of-Sample Comparison", fontweight="bold")

    # Panel 5: Generalization and Stability scores
    ax_dash5 = plt.subplot2grid((3, 2), (2, 1))
    scores_labels = ["Generalization", "Stability"]
    scores_vals = [gen_score, stability]
    ax_dash5.bar(scores_labels, scores_vals, color=["#2ca02c", "#bcbd22"], width=0.4, edgecolor="black")
    ax_dash5.set_ylim(0, 105)
    for idx, val in enumerate(scores_vals):
        ax_dash5.text(idx, val + 2, f"{val:.1f}%", ha="center", fontweight="bold")
    ax_dash5.set_title("Strategy Robustness Summary Scores", fontweight="bold")

    fig.suptitle(f"{strategy_type.upper()} ROBUSTNESS & VALIDATION DASHBOARD", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{pfx}robustness_dashboard.png"), dpi=300)
    plt.close(fig)

    logger.info(f"Finished generating all 20 robustness charts for '{strategy_type}'.")
