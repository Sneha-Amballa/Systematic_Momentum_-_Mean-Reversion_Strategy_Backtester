"""
Mean Reversion Strategy Signal Generator Module.

This module implements the core logic for the Rolling Z-Score Mean Reversion Strategy.
It handles signal generation via a finite-state machine (FSM), look-ahead bias prevention,
position mapping, trade detection, statistics generation, and professional visualizations.
"""

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.indicators import calculate_rolling_zscore, calculate_rolling_mean, calculate_rolling_std
from src.validators import validate_mean_reversion_params, validate_signals

logger = logging.getLogger(__name__)

class MeanReversionSignalGenerator:
    """
    Generates and analyzes trading signals for a rolling Z-Score mean reversion strategy.
    
    This is a Long-Only strategy. A Buy (Long) entry is triggered when the rolling Z-Score
    crosses below the Entry Threshold. The position is held until the rolling Z-Score
    crosses above the Exit Threshold.
    """

    def __init__(self, window: int = 20, entry_threshold: float = -2.0, exit_threshold: float = -0.5):
        """
        Initializes the MeanReversionSignalGenerator with strategy parameters.

        Args:
            window (int): The rolling lookback window size. Default is 20.
            entry_threshold (float): Z-Score threshold for entering a long position. Default is -2.0.
            exit_threshold (float): Z-Score threshold for exiting a long position. Default is -0.5.
        """
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
        """
        Computes rolling mean, std, z-scores, raw signals, execution signals, and positions.
        Ensures look-ahead bias prevention by shifting signals forward by 1 trading day.

        Args:
            df (pd.DataFrame): Preprocessed market data containing the close price.
            close_col (str): The column containing closing prices.

        Returns:
            pd.DataFrame: A copy of the DataFrame with new columns:
                - Rolling_Mean_<window>
                - Rolling_STD_<window>
                - Z_Score_<window>
                - Raw_Signal
                - Execution_Signal
                - Position
        """
        logger.info("Validating mean reversion indicators input parameters...")
        validate_mean_reversion_params(
            df=df,
            window=self.window,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            close_col=close_col
        )

        df_out = df.copy()

        # 1. Compute rolling indicators
        mean_col = f"Rolling_Mean_{self.window}"
        std_col = f"Rolling_STD_{self.window}"
        zscore_col = f"Z_Score_{self.window}"

        logger.info(f"Computing rolling indicators with window={self.window}...")
        df_out[mean_col] = calculate_rolling_mean(df_out[close_col], self.window)
        df_out[std_col] = calculate_rolling_std(df_out[close_col], self.window)
        df_out[zscore_col] = calculate_rolling_zscore(df_out[close_col], self.window)

        # 2. Raw Signal Generation: Finite-State Machine
        logger.info("Constructing raw mean reversion signals via FSM...")
        z_vals = df_out[zscore_col].values
        n = len(df_out)
        raw_signals = np.zeros(n)
        state = 0  # 0: Flat, 1: Long

        for i in range(n):
            z = z_vals[i]
            if np.isnan(z):
                raw_signals[i] = np.nan
                continue

            if state == 0:
                if z < self.entry_threshold:
                    state = 1
                    raw_signals[i] = 1.0
                else:
                    raw_signals[i] = 0.0
            elif state == 1:
                if z > self.exit_threshold:
                    state = 0
                    raw_signals[i] = 0.0
                else:
                    raw_signals[i] = 1.0

        df_out["Raw_Signal"] = raw_signals

        # 3. Look-Ahead Bias Prevention: Shift raw signals forward by 1 trading day
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
        Computes holding period for each trade.

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
                # Transition: 0 -> 1 (BUY Entry)
                in_position = True
                entry_date = date
                entry_idx = i
            elif in_position and pos == 0.0:
                # Transition: 1 -> 0 (SELL Exit)
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
        Computes summary statistics for mean reversion signals and positions.

        Args:
            df (pd.DataFrame): DataFrame with signals and positions.
            trades_df (pd.DataFrame): DataFrame of detected trades.

        Returns:
            Dict[str, Any]: Strategy summary statistics dictionary.
        """
        logger.info("Computing mean reversion strategy statistics...")
        
        # Filter warm-up rows
        active_df = df.dropna(subset=["Position"])
        total_active_days = len(active_df)

        if total_active_days == 0:
            logger.warning("No active trading period after warmup.")
            return {}

        zscore_col = f"Z_Score_{self.window}"

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

            # Z-Score statistics at entry and exit
            entry_z_trigger = []
            entry_z_exec = []
            exit_z_trigger = []
            exit_z_exec = []

            for _, trade in trades_df.iterrows():
                entry_dt = trade["Entry Date"]
                exit_dt = trade["Exit Date"]

                # Entry execution date index
                entry_loc = df.index.get_loc(entry_dt)
                if entry_loc > 0:
                    entry_z_trigger.append(df[zscore_col].iloc[entry_loc - 1])
                entry_z_exec.append(df[zscore_col].iloc[entry_loc])

                # Exit execution date index
                exit_loc = df.index.get_loc(exit_dt)
                if exit_loc > 0:
                    exit_z_trigger.append(df[zscore_col].iloc[exit_loc - 1])
                exit_z_exec.append(df[zscore_col].iloc[exit_loc])

            avg_entry_z_trigger = float(np.mean(entry_z_trigger)) if entry_z_trigger else np.nan
            avg_entry_z_exec = float(np.mean(entry_z_exec)) if entry_z_exec else np.nan
            avg_exit_z_trigger = float(np.mean(exit_z_trigger)) if exit_z_trigger else np.nan
            avg_exit_z_exec = float(np.mean(exit_z_exec)) if exit_z_exec else np.nan
        else:
            avg_hold_trading = max_hold_trading = min_hold_trading = 0
            avg_hold_calendar = max_hold_calendar = min_hold_calendar = 0
            avg_entry_z_trigger = avg_entry_z_exec = np.nan
            avg_exit_z_trigger = avg_exit_z_exec = np.nan

        # Transitions
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
            "Total_Active_Trading_Days": total_active_days,
            "Average_ZScore_Entry_Trigger": avg_entry_z_trigger,
            "Average_ZScore_Entry_Execution": avg_entry_z_exec,
            "Average_ZScore_Exit_Trigger": avg_exit_z_trigger,
            "Average_ZScore_Exit_Execution": avg_exit_z_exec
        }

        logger.info("Mean reversion summary statistics generated successfully.")
        return stats

    def plot_performance(self, df: pd.DataFrame, trades_df: pd.DataFrame, output_dir: str) -> None:
        """
        Generates and saves publication-quality charts for the mean reversion strategy.

        Figures created:
        1. Closing Price + Rolling Mean + Buy/Exit Markers (mean_reversion_price.png)
        2. Rolling Z-Score + Thresholds + Buy/Exit Markers (mean_reversion_zscore.png)
        3. Signal Timeline - Raw vs Shifted Execution Signal (mean_reversion_signal_timeline.png)
        4. Position Exposure Timeline (mean_reversion_position_timeline.png)
        5. Trade Duration distribution (mean_reversion_trade_durations.png)

        Args:
            df (pd.DataFrame): DataFrame containing price, rolling stats, signals, and position.
            trades_df (pd.DataFrame): DataFrame of detected trades.
            output_dir (str): Directory where charts will be saved.
        """
        logger.info(f"Generating professional visualizations. Saving to '{output_dir}'...")
        os.makedirs(output_dir, exist_ok=True)

        # Style configurations
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
        mean_col = f"Rolling_Mean_{self.window}"
        zscore_col = f"Z_Score_{self.window}"

        # ---------------------------------------------------------------------
        # FIGURE 1: Price and Rolling Mean with Buy/Exit Markers
        # ---------------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(14, 7), layout="constrained")
        ax1.plot(df.index, df[close_col], label="Nifty 50 Close", color="#1f77b4", linewidth=1.5, alpha=0.9)
        ax1.plot(df.index, df[mean_col], label=f"Rolling Mean ({self.window})", color="#ff7f0e", linewidth=1.2, linestyle="--")

        if not trades_df.empty:
            entries = df.loc[trades_df["Entry Date"]]
            ax1.scatter(
                entries.index, 
                entries[close_col], 
                marker="^", 
                color="#2ca02c", 
                s=100, 
                label="BUY Entry Execution", 
                zorder=5
            )
            # Only plot exits that are not terminal/open at end of dataset
            real_exits = trades_df[~trades_df["Closed_At_End"]]
            if not real_exits.empty:
                exits = df.loc[real_exits["Exit Date"]]
                ax1.scatter(
                    exits.index, 
                    exits[close_col], 
                    marker="v", 
                    color="#d62728", 
                    s=100, 
                    label="EXIT Execution", 
                    zorder=5
                )

        ax1.set_title(f"Nifty 50 Index & Rolling Mean (Window: {self.window})", fontweight="bold")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (INR)")
        ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        ax1.grid(True, linestyle=":", alpha=0.6)

        fig1_path = os.path.join(output_dir, "mean_reversion_price.png")
        fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
        plt.close(fig1)

        # ---------------------------------------------------------------------
        # FIGURE 2: Z-Score with Thresholds and Markers
        # ---------------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(14, 6), layout="constrained")
        ax2.plot(df.index, df[zscore_col], label="Z-Score", color="#9467bd", linewidth=1.2)
        ax2.axhline(self.entry_threshold, color="#d62728", linestyle="-.", linewidth=1.5, label=f"Entry Threshold ({self.entry_threshold})")
        ax2.axhline(self.exit_threshold, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Exit Threshold ({self.exit_threshold})")
        ax2.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

        if not trades_df.empty:
            entries = df.loc[trades_df["Entry Date"]]
            ax2.scatter(
                entries.index, 
                entries[zscore_col], 
                marker="^", 
                color="#2ca02c", 
                s=100, 
                label="BUY Entry Execution", 
                zorder=5
            )
            real_exits = trades_df[~trades_df["Closed_At_End"]]
            if not real_exits.empty:
                exits = df.loc[real_exits["Exit Date"]]
                ax2.scatter(
                    exits.index, 
                    exits[zscore_col], 
                    marker="v", 
                    color="#d62728", 
                    s=100, 
                    label="EXIT Execution", 
                    zorder=5
                )

        ax2.set_title(f"Rolling Z-Score Timeline (Window: {self.window})", fontweight="bold")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Z-Score")
        ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        ax2.grid(True, linestyle=":", alpha=0.6)

        fig2_path = os.path.join(output_dir, "mean_reversion_zscore.png")
        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)

        # ---------------------------------------------------------------------
        # FIGURE 3: Signal Timeline (Raw vs Execution Signal)
        # ---------------------------------------------------------------------
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
        
        ax3a.step(df.index, df["Raw_Signal"], where="pre", label="Raw Signal (t)", color="#7f7f7f", linewidth=1.5)
        ax3a.fill_between(df.index, df["Raw_Signal"], step="pre", color="#7f7f7f", alpha=0.1)
        ax3a.set_ylabel("Raw Signal")
        ax3a.set_ylim(-0.1, 1.1)
        ax3a.set_title("Strategy Signals: Raw (Today's Close) vs. Shifted Execution (Tomorrow)", fontweight="bold")
        ax3a.legend(loc="upper right")
        ax3a.grid(True, linestyle=":", alpha=0.6)

        ax3b.step(df.index, df["Execution_Signal"], where="pre", label="Execution Signal (t+1)", color="#17becf", linewidth=1.5)
        ax3b.fill_between(df.index, df["Execution_Signal"], step="pre", color="#17becf", alpha=0.1)
        ax3b.set_ylabel("Execution Signal")
        ax3b.set_xlabel("Date")
        ax3b.set_ylim(-0.1, 1.1)
        ax3b.legend(loc="upper right")
        ax3b.grid(True, linestyle=":", alpha=0.6)

        fig3_path = os.path.join(output_dir, "mean_reversion_signal_timeline.png")
        fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
        plt.close(fig3)

        # ---------------------------------------------------------------------
        # FIGURE 4: Position Exposure Timeline
        # ---------------------------------------------------------------------
        fig4, ax4 = plt.subplots(figsize=(14, 4), layout="constrained")
        ax4.step(df.index, df["Position"], where="pre", label="Portfolio Position State", color="#e377c2", linewidth=1.5)
        ax4.fill_between(df.index, df["Position"], step="pre", color="#e377c2", alpha=0.15)
        
        ax4.set_title("Portfolio Market Exposure (1 = Long, 0 = Flat)", fontweight="bold")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Position State")
        ax4.set_ylim(-0.1, 1.1)
        ax4.legend(loc="upper right")
        ax4.grid(True, linestyle=":", alpha=0.6)

        fig4_path = os.path.join(output_dir, "mean_reversion_position_timeline.png")
        fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
        plt.close(fig4)

        # ---------------------------------------------------------------------
        # FIGURE 5: Trade Duration Distribution
        # ---------------------------------------------------------------------
        fig5, ax5 = plt.subplots(figsize=(10, 6), layout="constrained")
        
        if not trades_df.empty:
            n_bins = min(15, len(trades_df))
            ax5.hist(
                trades_df["Holding Period (Trading Days)"], 
                bins=n_bins, 
                color="#bcbd22", 
                edgecolor="black", 
                alpha=0.75, 
                rwidth=0.85
            )
            avg_hold = trades_df["Holding Period (Trading Days)"].mean()
            ax5.axvline(avg_hold, color="red", linestyle="dashed", linewidth=1.5, label=f"Average Hold: {avg_hold:.1f} days")
            
            ax5.set_title("Distribution of Mean Reversion Trade Holding Periods", fontweight="bold")
            ax5.set_xlabel("Holding Period (Trading Days)")
            ax5.set_ylabel("Frequency (Count)")
            ax5.legend(loc="upper right")
        else:
            ax5.text(0.5, 0.5, "No Trades to Visualize", horizontalalignment="center", verticalalignment="center", fontsize=14)
            ax5.set_title("Distribution of Mean Reversion Trade Holding Periods", fontweight="bold")
        
        ax5.grid(True, linestyle=":", alpha=0.6)

        fig5_path = os.path.join(output_dir, "mean_reversion_trade_durations.png")
        fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")
        plt.close(fig5)

        logger.info("All mean reversion plots saved successfully in the output directory.")
