"""
Validation and Dataset Splitting Module.

Provides functions to split price series chronologically and perform
reusable grid-search parameter optimization on In-Sample periods.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.engine import backtest
from src.metrics import calculate_return_metrics, calculate_risk_adjusted_ratios
from src.risk import calculate_maximum_drawdown

logger = logging.getLogger(__name__)

def split_dataset(
    df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Splits the dataset chronologically according to start and end dates.

    Checks:
    - Dates are in chronological order.
    - Resulting DataFrame is not empty.
    - Input DataFrame contains a DatetimeIndex.

    Args:
        df (pd.DataFrame): Input market data DataFrame.
        start_date (str): Start date string (YYYY-MM-DD).
        end_date (str): End date string (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Sliced copy of the DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if start_ts > end_ts:
        raise ValueError(f"Start date {start_date} is after end date {end_date}.")

    sliced_df = df.loc[start_ts:end_ts].copy()

    if sliced_df.empty:
        raise ValueError(f"No observations found between {start_date} and {end_date}.")

    # Confirm chronological order
    if not sliced_df.index.is_monotonic_increasing:
        sliced_df = sliced_df.sort_index()

    logger.info(f"Split dataset successfully: {start_date} to {end_date} ({len(sliced_df)} observations).")
    return sliced_df

def optimize_parameters(
    prices: pd.DataFrame,
    strategy_type: str,
    param_grid: List[Dict[str, Any]],
    objective: str = "max_sharpe",
    initial_capital: float = 100000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    return_type: str = "simple",
    execution_type: str = "next_open"
) -> pd.DataFrame:
    """
    Performs grid search parameter optimization on the given price dataset.

    Args:
        prices (pd.DataFrame): Price data (Open, High, Low, Close).
        strategy_type (str): Either 'momentum' or 'mean_reversion'.
        param_grid (List[Dict[str, Any]]): List of parameter dictionaries to evaluate.
        objective (str): Metric to maximize/minimize. Supported:
            - 'max_sharpe': Sharpe Ratio (default)
            - 'max_cagr': CAGR
            - 'min_drawdown': Maximum Drawdown (minimized)
            - 'max_calmar': Calmar Ratio
            - 'max_sortino': Sortino Ratio
        initial_capital (float): Initial capital for the simulation.
        transaction_cost_bps (float): Transaction cost in bps.
        slippage_bps (float): Slippage in bps.
        return_type (str): Returns style ('simple' or 'log').
        execution_type (str): Execution lag style ('next_open' or 'next_close').

    Returns:
        pd.DataFrame: DataFrame containing all parameters and their performance metrics, ranked.
    """
    logger.info(f"Initializing parameter optimization sweep for '{strategy_type}'...")
    
    # Import generators dynamically to avoid circular dependencies
    if strategy_type == "momentum":
        from src.momentum import MomentumSignalGenerator as Generator
    elif strategy_type == "mean_reversion":
        from src.mean_reversion import MeanReversionSignalGenerator as Generator
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")

    results = []

    for i, params in enumerate(param_grid):
        logger.debug(f"Evaluating parameter set {i+1}/{len(param_grid)}: {params}")
        
        try:
            # 1. Instantiate signal generator and generate signals
            if strategy_type == "momentum":
                generator = Generator(
                    short_window=params["short_window"],
                    long_window=params["long_window"]
                )
            else: # mean_reversion
                generator = Generator(
                    window=params["window"],
                    entry_threshold=params["entry_threshold"],
                    exit_threshold=params["exit_threshold"]
                )

            signals_df = generator.generate_signals(prices)
            
            # 2. Run backtest simulation
            res = backtest(
                prices=prices,
                signals=signals_df["Raw_Signal"],
                initial_capital=initial_capital,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                return_type=return_type,
                execution_type=execution_type
            )
            
            # 3. Calculate metrics
            ret_metrics = calculate_return_metrics(res.portfolio["Portfolio_Value"], res.portfolio["Daily_Return"])
            risk_ratios = calculate_risk_adjusted_ratios(res.portfolio["Daily_Return"], res.portfolio["Portfolio_Value"])
            max_dd, _, _, _ = calculate_maximum_drawdown(res.portfolio["Portfolio_Value"])
            
            # Combine parameter values and metrics
            row = {**params}
            row["CAGR"] = ret_metrics.get("CAGR", 0.0)
            row["Volatility"] = res.portfolio["Daily_Return"].std(ddof=1) * np.sqrt(252.0)
            row["Sharpe_Ratio"] = risk_ratios.get("Sharpe_Ratio", 0.0)
            row["Sortino_Ratio"] = risk_ratios.get("Sortino_Ratio", 0.0)
            row["Calmar_Ratio"] = risk_ratios.get("Calmar_Ratio", 0.0)
            row["Max_Drawdown"] = max_dd * 100.0  # as percentage
            row["Total_Trades"] = res.summary["Total_Trades"]
            row["Total_Cost_Cash"] = res.summary["Total_Cost_Cash"]
            row["Total_Turnover"] = res.summary["Total_Turnover"]
            
            results.append(row)
        except Exception as e:
            logger.warning(f"Failed to evaluate parameter set {params}: {str(e)}", exc_info=True)

    if not results:
        raise RuntimeError("Parameter sweep failed to evaluate any combinations successfully.")

    df_results = pd.DataFrame(results)

    # Rank results according to objective
    # min_drawdown should sort ascending, others descending
    if objective == "max_sharpe":
        df_results = df_results.sort_values(by="Sharpe_Ratio", ascending=False)
    elif objective == "max_cagr":
        df_results = df_results.sort_values(by="CAGR", ascending=False)
    elif objective == "min_drawdown":
        df_results = df_results.sort_values(by="Max_Drawdown", ascending=True)
    elif objective == "max_calmar":
        df_results = df_results.sort_values(by="Calmar_Ratio", ascending=False)
    elif objective == "max_sortino":
        df_results = df_results.sort_values(by="Sortino_Ratio", ascending=False)
    else:
        raise ValueError(f"Unsupported optimization objective: {objective}")

    # Add Rank column
    df_results.insert(0, "Rank", range(1, len(df_results) + 1))
    
    logger.info(f"Parameter optimization sweep completed. Evaluated {len(df_results)} combinations.")
    return df_results
