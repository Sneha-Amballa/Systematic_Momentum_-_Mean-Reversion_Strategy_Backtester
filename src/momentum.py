"""
Momentum Strategy Signal Generator Module.

This module implements the core logic for the Moving Average Crossover Momentum Strategy.
It handles signal generation, look-ahead bias prevention, position mapping,
trade extraction, and performance statistics generation.
"""

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.indicators import calculate_sma
from src.validators import validate_indicator_params, validate_signals

logger = logging.getLogger(__name__)

class MomentumSignalGenerator:
    """
    Generates and analyzes trading signals for a moving average crossover strategy.
    
    This is a Long-Only strategy. A Buy (Long) signal is generated when the short
    moving average is strictly greater than the long moving average. No short-selling
    signals are generated.
    """

    def __init__(self, short_window: int = 20, long_window: int = 100):
        """
        Initializes the MomentumSignalGenerator with lookback windows.

        Args:
            short_window (int): The window size for the short moving average. Default is 20.
            long_window (int): The window size for the long moving average. Default is 100.
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
        """
        Computes SMAs, raw signals, execution signals, and positions.
        Ensures look-ahead bias prevention by shifting signals forward by 1 trading day.

        Args:
            df (pd.DataFrame): Preprocessed market data containing the close price.
            close_col (str): The column containing closing prices.

        Returns:
            pd.DataFrame: A copy of the DataFrame with new columns:
                - SMA_<short_window>
                - SMA_<long_window>
                - Raw_Signal
                - Execution_Signal
                - Position
        """
        logger.info("Validating indicators input parameters...")
        validate_indicator_params(df, self.short_window, self.long_window, close_col)

        df_out = df.copy()

        # 1. Calculate Simple Moving Averages
        short_sma_col = f"SMA_{self.short_window}"
        long_sma_col = f"SMA_{self.long_window}"
        
        logger.info(f"Computing moving averages: {short_sma_col} and {long_sma_col}")
        df_out[short_sma_col] = calculate_sma(df_out[close_col], self.short_window)
        df_out[long_sma_col] = calculate_sma(df_out[close_col], self.long_window)

        # 2. Raw Signal Generation: 1 if Short SMA > Long SMA, else 0
        # The raw signal is computed at the close of today.
        logger.info("Constructing raw crossover signals...")
        df_out["Raw_Signal"] = np.where(
            df_out[short_sma_col] > df_out[long_sma_col], 1.0, 0.0
        )
        
        # Apply NaN values to the warm-up lookback period (before long SMA is valid)
        warmup_mask = df_out[long_sma_col].isna()
        df_out.loc[warmup_mask, "Raw_Signal"] = np.nan

        # 3. Look-Ahead Bias Prevention: Shift raw signals forward by 1 trading day
        # Today's closing price signal is executed on tomorrow's trade.
        logger.info("Applying look-ahead bias shift (Execution Signal = Raw Signal shifted by 1)...")
        df_out["Execution_Signal"] = df_out["Raw_Signal"].shift(1)

        # 4. Position Construction
        # Position is identical to the execution signal for a long-only strategy
        logger.info("Constructing portfolio position columns...")
        df_out["Position"] = df_out["Execution_Signal"]

        # Run signal validator
        logger.info("Running signal validations...")
        validate_signals(df_out, "Raw_Signal", "Execution_Signal", "Position")
        logger.info("Signal validation completed successfully.")

        return df_out

    def detect_trades(self, df: pd.DataFrame, position_col: str = "Position") -> pd.DataFrame:
        """
        Identifies all BUY and SELL transition dates, mapping them to distinct trades.
        Computes the holding period for each trade.

        Args:
            df (pd.DataFrame): DataFrame with strategy signals and Position column.
            position_col (str): The column containing target position states.

        Returns:
            pd.DataFrame: A DataFrame containing identified trades, with columns:
                - Trade_ID: Unique identifier for each trade
                - Entry Date: Timestamp of entry into position
                - Exit Date: Timestamp of exit from position
                - Holding Period (Trading Days): Number of trading days in trade
                - Holding Period (Calendar Days): Number of calendar days in trade
                - Closed_At_End: Boolean indicating if position was closed because of dataset termination
        """
        logger.info("Detecting buy/sell trade events from position transitions...")
        trades = []
        trade_id = 0
        in_position = False
        entry_date = None
        entry_idx = None

        pos_series = df[position_col]

        for i in range(len(df)):
            pos = pos_series.iloc[i]
            date = df.index[i]

            if pd.isna(pos):
                continue

            if not in_position and pos == 1.0:
                # Transition: 0 -> 1 (BUY)
                in_position = True
                entry_date = date
                entry_idx = i
            elif in_position and pos == 0.0:
                # Transition: 1 -> 0 (SELL / Exit)
                in_position = False
                trade_id += 1
                exit_date = date
                holding_trading = i - entry_idx
                holding_calendar = (exit_date - entry_date).days
                trades.append({
                    "Trade_ID": trade_id,
                    "Entry Date": entry_date,
                    "Exit Date": exit_date,
                    "Holding Period (Trading Days)": holding_trading,
                    "Holding Period (Calendar Days)": holding_calendar,
                    "Closed_At_End": False
                })
                entry_date = None
                entry_idx = None

        # Handle terminal open position
        if in_position:
            trade_id += 1
            exit_date = df.index[-1]
            holding_trading = len(df) - 1 - entry_idx
            holding_calendar = (exit_date - entry_date).days
            trades.append({
                "Trade_ID": trade_id,
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Holding Period (Trading Days)": holding_trading,
                "Holding Period (Calendar Days)": holding_calendar,
                "Closed_At_End": True
            })
            logger.info(f"Open terminal trade detected. Closed simulated trade at final date: {exit_date.strftime('%Y-%m-%d')}")

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            logger.warning("No trades were detected in the signal set.")
            # Create empty df with schema
            trades_df = pd.DataFrame(columns=[
                "Trade_ID", "Entry Date", "Exit Date", 
                "Holding Period (Trading Days)", "Holding Period (Calendar Days)", "Closed_At_End"
            ])
        else:
            trades_df["Trade_ID"] = trades_df["Trade_ID"].astype(int)
            logger.info(f"Successfully extracted {len(trades_df)} complete trade cycles.")

        return trades_df

    def compute_statistics(self, df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes summary statistics for strategy signals and positions.

        Args:
            df (pd.DataFrame): DataFrame with signals and positions.
            trades_df (pd.DataFrame): DataFrame of detected trades.

        Returns:
            Dict[str, Any]: Strategy summary statistics dictionary.
        """
        logger.info("Computing strategy summary statistics...")
        
        # Filter warm-up rows to calculate accurate active metrics
        active_df = df.dropna(subset=["Position"])
        total_active_days = len(active_df)

        if total_active_days == 0:
            logger.warning("No active trading period after warmup.")
            return {}

        # 1. Total Signals
        raw_buys = int((df["Raw_Signal"] == 1.0).sum())
        raw_sells = int((df["Raw_Signal"] == 0.0).sum())
        exec_buys = int((df["Execution_Signal"] == 1.0).sum())
        exec_sells = int((df["Execution_Signal"] == 0.0).sum())

        # 2. Position stats
        days_invested = int((active_df["Position"] == 1.0).sum())
        days_flat = int((active_df["Position"] == 0.0).sum())
        
        pct_invested = (days_invested / total_active_days) * 100
        pct_flat = (days_flat / total_active_days) * 100

        # 3. Trade stats
        num_trades = len(trades_df)
        
        if num_trades > 0:
            avg_hold_trading = float(trades_df["Holding Period (Trading Days)"].mean())
            max_hold_trading = int(trades_df["Holding Period (Trading Days)"].max())
            min_hold_trading = int(trades_df["Holding Period (Trading Days)"].min())

            avg_hold_calendar = float(trades_df["Holding Period (Calendar Days)"].mean())
            max_hold_calendar = int(trades_df["Holding Period (Calendar Days)"].max())
            min_hold_calendar = int(trades_df["Holding Period (Calendar Days)"].min())
        else:
            avg_hold_trading = max_hold_trading = min_hold_trading = 0
            avg_hold_calendar = max_hold_calendar = min_hold_calendar = 0

        # Total buy transitions = number of trades (each trade has an entry)
        # Total sell transitions = number of trades that were closed (not including open at end if we didn't count it,
        # but since we closed the last trade at the end, total exits matches total trades)
        total_buys = num_trades
        total_sells = len(trades_df[~trades_df["Closed_At_End"]])

        stats = {
            "Total_Raw_Buy_Days": raw_buys,
            "Total_Raw_Sell_Days": raw_sells,
            "Total_Execution_Buy_Days": exec_buys,
            "Total_Execution_Sell_Days": exec_sells,
            "Total_Buy_Signals_Transitions": total_buys,
            "Total_Sell_Signals_Transitions": total_sells,
            "Number_of_Trades": num_trades,
            "Average_Holding_Period_Trading_Days": avg_hold_trading,
            "Maximum_Holding_Period_Trading_Days": max_hold_trading,
            "Minimum_Holding_Period_Trading_Days": min_hold_trading,
            "Average_Holding_Period_Calendar_Days": avg_hold_calendar,
            "Maximum_Holding_Period_Calendar_Days": max_hold_calendar,
            "Minimum_Holding_Period_Calendar_Days": min_hold_calendar,
            "Longest_Continuous_Position_Trading_Days": max_hold_trading,
            "Percentage_of_Time_Invested": pct_invested,
            "Flat_Market_Percentage": pct_flat,
            "Total_Active_Trading_Days": total_active_days
        }

        logger.info("Summary statistics generated successfully.")
        return stats

    def plot_performance(
        self, df: pd.DataFrame, trades_df: pd.DataFrame, output_dir: str
    ) -> None:
        """
        Generates and saves publication-quality charts for the momentum strategy.

        Figures created:
        1. Closing Price + SMA20 + SMA100 + Entry/Exit Markers (price_sma_crossover.png)
        2. Signal Timeline - Raw vs Shifted Execution Signal (signal_timeline.png)
        3. Position Exposure Timeline (position_timeline.png)
        4. Trade Duration distribution (trade_durations.png)

        Args:
            df (pd.DataFrame): DataFrame containing price, SMAs, signals, and position.
            trades_df (pd.DataFrame): DataFrame of detected trades.
            output_dir (str): Directory where charts will be saved.
        """
        logger.info(f"Generating professional visualizations. Saving to '{output_dir}'...")
        os.makedirs(output_dir, exist_ok=True)

        # Apply high-quality style options
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

        close_col = "Close"
        short_sma_col = f"SMA_{self.short_window}"
        long_sma_col = f"SMA_{self.long_window}"

        # ---------------------------------------------------------------------
        # FIGURE 1: Price and SMA Crossover with Buy/Sell Markers
        # ---------------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(14, 7), layout="constrained")
        
        # Plot price and SMAs
        ax1.plot(df.index, df[close_col], label="Nifty 50 Close", color="#1f77b4", linewidth=1.5, alpha=0.9)
        ax1.plot(df.index, df[short_sma_col], label=f"Short SMA ({self.short_window})", color="#2ca02c", linewidth=1.2, linestyle="--")
        ax1.plot(df.index, df[long_sma_col], label=f"Long SMA ({self.long_window})", color="#d62728", linewidth=1.5)

        # Plot Buy (Green Up-Triangle) and Sell (Red Down-Triangle) markers
        if not trades_df.empty:
            # Entry points
            entries = df.loc[trades_df["Entry Date"]]
            ax1.scatter(
                entries.index, 
                entries[close_col], 
                marker="^", 
                color="#006400", 
                s=100, 
                label="BUY Entry Signal", 
                zorder=5
            )
            # Exit points
            exits = df.loc[trades_df["Exit Date"]]
            ax1.scatter(
                exits.index, 
                exits[close_col], 
                marker="v", 
                color="#8b0000", 
                s=100, 
                label="SELL Exit Signal", 
                zorder=5
            )

        ax1.set_title("Nifty 50 Index: Moving Average Crossover Strategy", fontweight="bold")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (INR)")
        ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        ax1.grid(True, linestyle=":", alpha=0.6)

        fig1_path = os.path.join(output_dir, "price_sma_crossover.png")
        fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
        plt.close(fig1)

        # ---------------------------------------------------------------------
        # FIGURE 2: Signal Timeline (Raw vs Execution Signal)
        # ---------------------------------------------------------------------
        # We will show the lag by zooming in on a small window (e.g. 150 trading days) 
        # to make the 1-day execution shift visibly clear, and show the whole timeline.
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
        
        # Subplot 1: Raw Signal
        ax2a.step(df.index, df["Raw_Signal"], where="pre", label="Raw Signal", color="#7f7f7f", linewidth=1.5)
        ax2a.fill_between(df.index, df["Raw_Signal"], step="pre", color="#7f7f7f", alpha=0.1)
        ax2a.set_ylabel("Raw Signal")
        ax2a.set_ylim(-0.1, 1.1)
        ax2a.set_title("Strategy Signals Timeline (Raw vs. Shifted Execution)", fontweight="bold")
        ax2a.legend(loc="upper right")
        ax2a.grid(True, linestyle=":", alpha=0.6)

        # Subplot 2: Execution Signal
        ax2b.step(df.index, df["Execution_Signal"], where="pre", label="Execution Signal (Shifted)", color="#17becf", linewidth=1.5)
        ax2b.fill_between(df.index, df["Execution_Signal"], step="pre", color="#17becf", alpha=0.1)
        ax2b.set_ylabel("Execution Signal")
        ax2b.set_xlabel("Date")
        ax2b.set_ylim(-0.1, 1.1)
        ax2b.legend(loc="upper right")
        ax2b.grid(True, linestyle=":", alpha=0.6)

        fig2_path = os.path.join(output_dir, "signal_timeline.png")
        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)

        # ---------------------------------------------------------------------
        # FIGURE 3: Position Exposure Timeline
        # ---------------------------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(14, 4), layout="constrained")
        ax3.step(df.index, df["Position"], where="pre", label="Market Exposure", color="#e377c2", linewidth=1.5)
        ax3.fill_between(df.index, df["Position"], step="pre", color="#e377c2", alpha=0.15)
        
        ax3.set_title("Strategy Position State (1 = Invested, 0 = Flat)", fontweight="bold")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Position State")
        ax3.set_ylim(-0.1, 1.1)
        ax3.legend(loc="upper right")
        ax3.grid(True, linestyle=":", alpha=0.6)

        fig3_path = os.path.join(output_dir, "position_timeline.png")
        fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
        plt.close(fig3)

        # ---------------------------------------------------------------------
        # FIGURE 4: Trade Duration Distribution
        # ---------------------------------------------------------------------
        fig4, ax4 = plt.subplots(figsize=(10, 6), layout="constrained")
        
        if not trades_df.empty:
            # We plot a histogram of holding periods
            n, bins, patches = ax4.hist(
                trades_df["Holding Period (Trading Days)"], 
                bins=min(15, len(trades_df)), 
                color="#bcbd22", 
                edgecolor="black", 
                alpha=0.75, 
                rwidth=0.85
            )
            # Add average indicator line
            avg_hold = trades_df["Holding Period (Trading Days)"].mean()
            ax4.axvline(avg_hold, color="red", linestyle="dashed", linewidth=1.5, label=f"Average Hold: {avg_hold:.1f} days")
            
            ax4.set_title("Distribution of Trade Holding Periods", fontweight="bold")
            ax4.set_xlabel("Holding Period (Trading Days)")
            ax4.set_ylabel("Frequency (Count)")
            ax4.legend(loc="upper right")
        else:
            ax4.text(0.5, 0.5, "No Trades to Visualize", horizontalalignment="center", verticalalignment="center", fontsize=14)
            ax4.set_title("Distribution of Trade Holding Periods", fontweight="bold")
        
        ax4.grid(True, linestyle=":", alpha=0.6)

        fig4_path = os.path.join(output_dir, "trade_durations.png")
        fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
        plt.close(fig4)

        logger.info("All plots saved successfully in the output directory.")
