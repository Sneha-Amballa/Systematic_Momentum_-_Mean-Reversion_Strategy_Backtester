"""
Portfolio Simulation and Return Calculations Module.

Provides pure mathematical functions to compute asset returns, raw strategy returns,
and daily portfolio value simulations vectorially.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def calculate_asset_returns(prices: pd.DataFrame, return_type: str = "simple", close_col: str = "Close") -> pd.Series:
    """
    Computes close-to-close returns of the underlying asset.

    Args:
        prices (pd.DataFrame): DataFrame containing price history.
        return_type (str): Type of returns: 'simple' or 'log'.
        close_col (str): Column name for closing prices.

    Returns:
        pd.Series: Asset returns series.
    """
    if close_col not in prices.columns:
        raise ValueError(f"prices DataFrame must contain closing column '{close_col}'")

    close = prices[close_col]
    if return_type == "simple":
        returns = close.pct_change().fillna(0.0)
        returns.name = "Asset_Simple_Return"
    elif return_type == "log":
        returns = np.log(close / close.shift(1)).fillna(0.0)
        returns.name = "Asset_Log_Return"
    else:
        raise ValueError(f"Unknown return_type: {return_type}. Must be 'simple' or 'log'.")

    return returns

def calculate_strategy_returns(
    prices: pd.DataFrame,
    position: pd.Series,
    execution_type: str = "next_open",
    return_type: str = "simple"
) -> pd.Series:
    """
    Calculates raw daily strategy returns vectorially based on positions and price transitions.

    Args:
        prices (pd.DataFrame): DataFrame containing Open and Close prices.
        position (pd.Series): Portfolio position states (1.0 for Long, -1.0 for Short, 0.0 for Flat).
        execution_type (str): Timing of execution ('next_open' or 'next_close').
        return_type (str): Return calculation method ('simple' or 'log').

    Returns:
        pd.Series: Daily strategy returns.
    """
    if "Close" not in prices.columns:
        raise ValueError("prices DataFrame must contain a 'Close' column.")

    close = prices["Close"]
    pos_prev = position.shift(1).fillna(0.0)

    logger.info(
        f"Computing strategy returns (execution_type='{execution_type}', return_type='{return_type}')"
    )

    if execution_type == "next_close":
        # Under next-close execution, signals from close t-1 execute at close t.
        # The return captured on day t is based on close-to-close returns, scaled by the position held.
        # Since position is shifted by 2, Position_t = Raw_Signal_{t-2}.
        if return_type == "simple":
            asset_ret = close.pct_change().fillna(0.0)
        elif return_type == "log":
            asset_ret = np.log(close / close.shift(1)).fillna(0.0)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")
        
        strategy_ret = position * asset_ret

    elif execution_type == "next_open":
        # Under next-open execution, signals from close t-1 execute at Open t.
        # We capture intraday return on entry days, close-to-close on hold days, and overnight on exit days.
        if "Open" not in prices.columns:
            raise ValueError("prices DataFrame must contain an 'Open' column for next_open execution.")
        
        open_px = prices["Open"]
        close_prev = close.shift(1).fillna(open_px)

        strategy_ret = pd.Series(0.0, index=position.index)

        # Decompose exposure transitions
        is_buy_entry = (pos_prev <= 0.0) & (position == 1.0)
        is_hold_long = (pos_prev == 1.0) & (position == 1.0)
        is_sell_exit = (pos_prev == 1.0) & (position == 0.0)

        is_short_entry = (pos_prev >= 0.0) & (position == -1.0)
        is_hold_short = (pos_prev == -1.0) & (position == -1.0)
        is_cover_exit = (pos_prev == -1.0) & (position == 0.0)

        is_long_to_short = (pos_prev == 1.0) & (position == -1.0)
        is_short_to_long = (pos_prev == -1.0) & (position == 1.0)

        if return_type == "simple":
            # Buy Entry: Return from Open_t to Close_t
            strategy_ret[is_buy_entry] = (close - open_px) / open_px
            # Hold Long: Return from Close_t-1 to Close_t
            strategy_ret[is_hold_long] = (close - close_prev) / close_prev
            # Sell Exit: Return from Close_t-1 to Open_t
            strategy_ret[is_sell_exit] = (open_px - close_prev) / close_prev

            # Short Entry: Return from Open_t to Close_t (short)
            strategy_ret[is_short_entry] = (open_px - close) / open_px
            # Hold Short: Return from Close_t-1 to Close_t (short)
            strategy_ret[is_hold_short] = (close_prev - close) / close_prev
            # Cover Exit: Return from Close_t-1 to Open_t (short)
            strategy_ret[is_cover_exit] = (close_prev - open_px) / close_prev

            # Position Flips (compounded)
            # Long to Short: Sell Long at Open + Short at Open
            strategy_ret[is_long_to_short] = (2.0 * open_px - close) / close_prev - 1.0
            # Short to Long: Cover Short at Open + Buy Long at Open
            strategy_ret[is_short_to_long] = ((2.0 * close_prev - open_px) / close_prev) * (close / open_px) - 1.0

        elif return_type == "log":
            r_intraday = np.log(close / open_px)
            r_overnight = np.log(open_px / close_prev)
            r_close_to_close = np.log(close / close_prev)

            strategy_ret[is_buy_entry] = r_intraday
            strategy_ret[is_hold_long] = r_close_to_close
            strategy_ret[is_sell_exit] = r_overnight

            strategy_ret[is_short_entry] = -r_intraday
            strategy_ret[is_hold_short] = -r_close_to_close
            strategy_ret[is_cover_exit] = -r_overnight

            # Additive log returns for flips
            strategy_ret[is_long_to_short] = r_overnight - r_intraday
            strategy_ret[is_short_to_long] = -r_overnight + r_intraday
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

    else:
        raise ValueError(f"Unknown execution_type: {execution_type}")

    strategy_ret.name = "Raw_Strategy_Return"
    return strategy_ret.fillna(0.0)

def simulate_portfolio(
    strategy_returns: pd.Series,
    initial_capital: float = 100000.0,
    return_type: str = "simple"
) -> pd.DataFrame:
    """
    Simulates portfolio equity curves and PnL daily vectorially.

    Args:
        strategy_returns (pd.Series): Daily net returns of the strategy.
        initial_capital (float): Starting balance. Default is 100000.0.
        return_type (str): Type of returns: 'simple' or 'log'.

    Returns:
        pd.DataFrame: Portfolio history containing:
            - Daily_Return: daily net return.
            - Cumulative_Return: total net return since start.
            - Portfolio_Value: daily equity balance.
            - Equity_Curve: identical to Portfolio_Value.
            - Daily_PnL: daily rupee change in value.
            - Portfolio_Growth: index of capital growth (Portfolio_Value / Initial_Capital).
    """
    df = pd.DataFrame(index=strategy_returns.index)
    df["Daily_Return"] = strategy_returns

    logger.info(f"Simulating portfolio value starting at ₹{initial_capital:,.2f} using {return_type} returns...")

    if return_type == "simple":
        df["Cumulative_Return"] = (1.0 + strategy_returns).cumprod() - 1.0
        df["Portfolio_Value"] = initial_capital * (1.0 + strategy_returns).cumprod()
    elif return_type == "log":
        df["Cumulative_Return"] = np.exp(strategy_returns.cumsum()) - 1.0
        df["Portfolio_Value"] = initial_capital * np.exp(strategy_returns.cumsum())
    else:
        raise ValueError(f"Unknown return_type: {return_type}")

    # Daily PnL calculation
    df["Daily_PnL"] = df["Portfolio_Value"].diff().fillna(df["Portfolio_Value"] - initial_capital)
    df["Equity_Curve"] = df["Portfolio_Value"]
    df["Portfolio_Growth"] = df["Portfolio_Value"] / initial_capital

    return df
