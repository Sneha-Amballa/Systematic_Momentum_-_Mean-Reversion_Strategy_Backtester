"""
Vectorized Backtesting Engine Core Module.

Provides a dataclass to hold backtesting results and the orchestration functions
to validate inputs, run the backtest, generate the trade book, and return results.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.costs import calculate_transaction_costs
from src.execution import map_signals_to_positions
from src.portfolio import calculate_strategy_returns, simulate_portfolio
from src.validators import validate_backtest_inputs, validate_backtest_results

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """
    Dataclass containing the outputs of a vectorized backtest run.
    """
    positions: pd.Series
    returns: pd.DataFrame
    trade_book: pd.DataFrame
    costs: pd.DataFrame
    portfolio: pd.DataFrame
    equity_curve: pd.Series
    summary: Dict[str, Any]

def generate_trade_book(
    prices: pd.DataFrame,
    position: pd.Series,
    cost_bps: float,
    slippage_bps: float,
    initial_capital: float,
    execution_type: str = "next_open"
) -> pd.DataFrame:
    """
    Generates the trade history book from position transitions.

    Args:
        prices (pd.DataFrame): Market prices with Open and Close.
        position (pd.Series): Portfolio positions.
        cost_bps (float): Transaction cost in basis points.
        slippage_bps (float): Slippage in basis points.
        initial_capital (float): Starting balance.
        execution_type (str): Timing of execution.

    Returns:
        pd.DataFrame: Trade history records.
    """
    logger.info("Reconstructing trade book from position transitions...")

    pos_prev = position.shift(1).fillna(0.0)
    trades = []
    active_trade = None
    trade_id = 0

    cost_rate = cost_bps / 10000.0
    slippage_rate = slippage_bps / 10000.0

    for i in range(len(position)):
        date = position.index[i]
        pos = position.iloc[i]
        prev_pos = pos_prev.iloc[i]

        if pos == prev_pos:
            continue

        # Close active position if exposure changes or flips
        if active_trade is not None:
            # Check if we are reducing exposure or flipping sign
            if (active_trade["Position Size"] > 0 and pos < active_trade["Position Size"]) or \
               (active_trade["Position Size"] < 0 and pos > active_trade["Position Size"]):
                
                # Determine exit execution price
                if execution_type == "next_open":
                    exit_price = prices.loc[date, "Open"]
                else:
                    exit_price = prices["Close"].iloc[i - 1]

                active_trade["Exit Date"] = date
                active_trade["Exit Price"] = exit_price
                active_trade["Holding Days"] = i - position.index.get_loc(active_trade["Entry Date"])
                
                # Exit cost in capital space
                exit_value = initial_capital * abs(active_trade["Position Size"])
                active_trade["Exit Cost"] = exit_value * cost_rate
                active_trade["Exit Slippage"] = exit_value * slippage_rate
                active_trade["Total Cost"] += active_trade["Exit Cost"] + active_trade["Exit Slippage"]

                trades.append(active_trade)
                active_trade = None

        # Open new position if non-zero
        if pos != 0.0:
            if execution_type == "next_open":
                entry_price = prices.loc[date, "Open"]
            else:
                entry_price = prices["Close"].iloc[i - 1]

            trade_id += 1
            entry_value = initial_capital * abs(pos)
            entry_cost = entry_value * cost_rate
            entry_slip = entry_value * slippage_rate

            active_trade = {
                "Trade ID": trade_id,
                "Entry Date": date,
                "Exit Date": None,
                "Holding Days": 0,
                "Entry Price": entry_price,
                "Exit Price": None,
                "Position Size": pos,
                "Entry Cost": entry_cost,
                "Entry Slippage": entry_slip,
                "Exit Cost": 0.0,
                "Exit Slippage": 0.0,
                "Total Cost": entry_cost + entry_slip
            }

    # Handle open positions at the end of the simulation
    if active_trade is not None:
        final_date = position.index[-1]
        active_trade["Exit Date"] = final_date
        active_trade["Exit Price"] = prices.loc[final_date, "Close"]
        active_trade["Holding Days"] = len(position) - 1 - position.index.get_loc(active_trade["Entry Date"])
        active_trade["Exit Cost"] = initial_capital * abs(active_trade["Position Size"]) * cost_rate
        active_trade["Exit Slippage"] = initial_capital * abs(active_trade["Position Size"]) * slippage_rate
        active_trade["Total Cost"] += active_trade["Exit Cost"] + active_trade["Exit Slippage"]
        active_trade["Closed_At_End"] = True
        trades.append(active_trade)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        logger.warning("No completed trades detected in simulation.")
        trades_df = pd.DataFrame(columns=[
            "Trade ID", "Entry Date", "Exit Date", "Holding Days",
            "Entry Price", "Exit Price", "Position Size",
            "Entry Cost", "Entry Slippage", "Exit Cost", "Exit Slippage", "Total Cost"
        ])
    else:
        # Standardize types
        trades_df["Trade ID"] = trades_df["Trade ID"].astype(int)
        trades_df["Holding Days"] = trades_df["Holding Days"].astype(int)
        if "Closed_At_End" not in trades_df.columns:
            trades_df["Closed_At_End"] = False
        else:
            trades_df["Closed_At_End"] = trades_df["Closed_At_End"].fillna(False)

    return trades_df

def backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    return_type: str = "simple",
    execution_type: str = "next_open",
    apply_cost_on: str = "both"
) -> BacktestResult:
    """
    Orchestrates the running of a vectorized backtest simulation.

    Args:
        prices (pd.DataFrame): Market data (Open, Close columns).
        signals (pd.Series): Binary strategy signal history.
        initial_capital (float): Starting balance. Default is 100000.0.
        transaction_cost_bps (float): Transaction cost in basis points. Default is 5.0.
        slippage_bps (float): Slippage in basis points. Default is 2.0.
        return_type (str): Return calculation type: 'simple' or 'log'.
        execution_type (str): timing timing rules: 'next_open' or 'next_close'.
        apply_cost_on (str): Cost type trigger: 'entry', 'exit', or 'both'.

    Returns:
        BacktestResult: Dataclass containing the simulation results.
    """
    # 1. Input validation
    validate_backtest_inputs(prices, signals)

    # 2. Map signals to positions
    position = map_signals_to_positions(signals, execution_type)

    # 3. Calculate daily strategy raw returns
    raw_return = calculate_strategy_returns(prices, position, execution_type, return_type)

    # 4. Calculate transaction costs and slippage daily returns impact
    tx_cost_ret, slippage_ret, total_cost_ret = calculate_transaction_costs(
        position, transaction_cost_bps, slippage_bps, apply_cost_on
    )

    # 5. Compute net returns
    net_return = raw_return - total_cost_ret
    # Ensure first day return is 0.0 to prevent look-ahead start spikes
    net_return.iloc[0] = 0.0

    # 6. Simulate portfolio capital curves
    portfolio_df = simulate_portfolio(net_return, initial_capital, return_type)

    # Validate output
    validate_backtest_results(portfolio_df, initial_capital)

    # 7. Generate trade history book
    trade_book = generate_trade_book(
        prices, position, transaction_cost_bps, slippage_bps, initial_capital, execution_type
    )

    # Construct auxiliary dataframes
    returns_df = pd.DataFrame({
        "Raw_Return": raw_return,
        "Net_Return": net_return
    }, index=position.index)

    costs_df = pd.DataFrame({
        "Transaction_Cost_Return": tx_cost_ret,
        "Slippage_Return": slippage_ret,
        "Total_Cost_Return": total_cost_ret
    }, index=position.index)

    # Summary statistics dict
    total_trades = len(trade_book)
    turnover = (position.diff().fillna(0.0).abs()).sum()
    total_cost_cash = (portfolio_df["Portfolio_Value"] * total_cost_ret).sum()

    summary = {
        "Initial_Capital": initial_capital,
        "Final_Capital": portfolio_df["Portfolio_Value"].iloc[-1],
        "Total_Return_Pct": portfolio_df["Cumulative_Return"].iloc[-1] * 100.0,
        "Total_Trades": total_trades,
        "Total_Turnover": turnover,
        "Total_Cost_Cash": total_cost_cash,
        "Average_Hold_Days": float(trade_book["Holding Days"].mean()) if total_trades > 0 else 0.0
    }

    logger.info("Vectorized backtest simulation completed successfully.")

    return BacktestResult(
        positions=position,
        returns=returns_df,
        trade_book=trade_book,
        costs=costs_df,
        portfolio=portfolio_df,
        equity_curve=portfolio_df["Portfolio_Value"],
        summary=summary
    )

def plot_backtest_results(
    result: BacktestResult,
    prices: pd.DataFrame,
    output_dir: str,
    prefix: str = ""
) -> None:
    """
    Generates 10 publication-quality backtest charts and saves them to the output directory.

    Plots generated:
    1. Portfolio Value over time (portfolio_value.png)
    2. Equity Curve vs Nifty 50 benchmark (equity_curve.png)
    3. Daily Portfolio Returns timeline (daily_returns.png)
    4. Daily PnL rupee timeline (daily_pnl.png)
    5. Transaction Cost Return Impact timeline (transaction_cost_timeline.png)
    6. Slippage Return Impact timeline (slippage_timeline.png)
    7. Trade Holding Period Distribution (trade_distribution.png)
    8. Capital Growth Factor timeline (capital_growth.png)
    9. Exposure Exposure Timeline (exposure_timeline.png)
    10. Daily Portfolio Turnover Timeline (turnover_timeline.png)

    Args:
        result (BacktestResult): Backtest output dataclass.
        prices (pd.DataFrame): DataFrame containing underlying price index history (Close).
        output_dir (str): Folder where files will be written.
        prefix (str): File prefix (e.g. 'momentum' or 'mean_reversion').
    """
    import os
    import matplotlib.pyplot as plt

    logger.info(f"Generating 10 portfolio charts for '{prefix}' in '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 300
    })

    pfx = f"{prefix}_" if prefix else ""

    # 1. Portfolio Value Timeline
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(result.portfolio.index, result.portfolio["Portfolio_Value"], color="#1f77b4", linewidth=1.5)
    ax.set_title(f"{prefix.capitalize()} Strategy: Portfolio Value", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital (INR)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}portfolio_value.png"), dpi=300)
    plt.close(fig)

    # 2. Equity Curve comparison vs Benchmark (Nifty 50 close price growth)
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    benchmark = prices["Close"] / prices["Close"].iloc[0] * result.summary["Initial_Capital"]
    ax.plot(result.equity_curve.index, result.equity_curve, label="Strategy Equity Curve", color="#1f77b4", linewidth=1.8)
    ax.plot(benchmark.index, benchmark, label="Nifty 50 Buy & Hold Benchmark", color="#7f7f7f", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_title(f"{prefix.capitalize()} Strategy: Equity Curve vs Benchmark", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (INR)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}equity_curve.png"), dpi=300)
    plt.close(fig)

    # 3. Daily Portfolio Returns
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(result.portfolio.index, result.portfolio["Daily_Return"], color="#2ca02c", linewidth=0.8, alpha=0.7)
    ax.set_title(f"{prefix.capitalize()} Strategy: Daily Portfolio Returns", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}daily_returns.png"), dpi=300)
    plt.close(fig)

    # 4. Daily PnL Timeline
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(result.portfolio.index, result.portfolio["Daily_PnL"], color="#d62728", linewidth=0.8, alpha=0.7)
    ax.set_title(f"{prefix.capitalize()} Strategy: Daily Portfolio PnL (INR)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL (INR)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}daily_pnl.png"), dpi=300)
    plt.close(fig)

    # 5. Transaction Cost Timeline (daily returns impact)
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    non_zero_costs = result.costs["Transaction_Cost_Return"][result.costs["Transaction_Cost_Return"] > 0]
    if not non_zero_costs.empty:
        ax.bar(non_zero_costs.index, non_zero_costs * 10000.0, color="#ff7f0e", width=5.0)
    ax.set_title(f"{prefix.capitalize()} Strategy: Daily Transaction Costs Paid", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost (Basis Points)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}transaction_cost_timeline.png"), dpi=300)
    plt.close(fig)

    # 6. Slippage Timeline (daily returns impact)
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    non_zero_slip = result.costs["Slippage_Return"][result.costs["Slippage_Return"] > 0]
    if not non_zero_slip.empty:
        ax.bar(non_zero_slip.index, non_zero_slip * 10000.0, color="#e377c2", width=5.0)
    ax.set_title(f"{prefix.capitalize()} Strategy: Daily Slippage Paid", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Slippage (Basis Points)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}slippage_timeline.png"), dpi=300)
    plt.close(fig)

    # 7. Trade Distribution (histogram of trade holding periods)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    if not result.trade_book.empty:
        ax.hist(result.trade_book["Holding Days"], bins=min(15, len(result.trade_book)), color="#bcbd22", edgecolor="black", alpha=0.75, rwidth=0.85)
        avg_hold = result.trade_book["Holding Days"].mean()
        ax.axvline(avg_hold, color="red", linestyle="dashed", linewidth=1.5, label=f"Average Hold: {avg_hold:.1f} Days")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No Trades to Plot", ha="center", va="center")
    ax.set_title(f"{prefix.capitalize()} Strategy: Holding Period Distribution", fontweight="bold")
    ax.set_xlabel("Holding Period (Trading Days)")
    ax.set_ylabel("Frequency (Count)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}trade_distribution.png"), dpi=300)
    plt.close(fig)

    # 8. Capital Growth (Portfolio Growth)
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(result.portfolio.index, result.portfolio["Portfolio_Growth"], color="#17becf", linewidth=1.5)
    ax.set_title(f"{prefix.capitalize()} Strategy: Capital Growth Factor", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Factor (Start = 1.0)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}capital_growth.png"), dpi=300)
    plt.close(fig)

    # 9. Exposure Timeline
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.step(result.positions.index, result.positions, where="pre", color="#9467bd", linewidth=1.2)
    ax.fill_between(result.positions.index, result.positions, step="pre", color="#9467bd", alpha=0.15)
    ax.set_title(f"{prefix.capitalize()} Strategy: Portfolio Exposure Timeline", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure State")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}exposure_timeline.png"), dpi=300)
    plt.close(fig)

    # 10. Turnover Timeline
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    daily_turnover = result.positions.diff().fillna(0.0).abs()
    non_zero_turnover = daily_turnover[daily_turnover > 0]
    if not non_zero_turnover.empty:
        ax.bar(non_zero_turnover.index, non_zero_turnover, color="#17becf", width=5.0)
    ax.set_title(f"{prefix.capitalize()} Strategy: Daily Portfolio Turnover Timeline", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover (Units)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(os.path.join(output_dir, f"{pfx}turnover_timeline.png"), dpi=300)
    plt.close(fig)

    logger.info(f"Finished generating all 10 visualizations with prefix '{prefix}'.")

