"""
Risk Metrics and Downside Exposure Module.

Provides mathematical risk analytics functions, including volatility, downside deviation,
drawdown metrics, Ulcer Index, Semi Deviation, and Value at Risk (VaR/CVaR).
"""

import logging
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def calculate_volatility(returns: pd.Series, annualization_factor: float = 252.0) -> float:
    """
    Calculates annualized volatility of returns.

    Args:
        returns (pd.Series): Series of strategy returns.
        annualization_factor (float): Number of trading periods in a year. Default is 252.0.

    Returns:
        float: Annualized volatility.
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(annualization_factor))

def calculate_downside_volatility(
    returns: pd.Series,
    target_return: float = 0.0,
    annualization_factor: float = 252.0
) -> float:
    """
    Calculates annualized downside volatility (downside deviation).

    Unlike regular standard deviation, downside deviation only penalizes returns
    that fall below a target return (often zero or the risk-free rate).

    Formula:
        sigma_down = sqrt( (1 / N) * sum(min(0, R_t - target)^2) ) * sqrt(annualization_factor)

    Args:
        returns (pd.Series): Series of strategy returns.
        target_return (float): Minimum acceptable return. Default is 0.0.
        annualization_factor (float): Number of trading periods in a year. Default is 252.0.

    Returns:
        float: Annualized downside deviation.
    """
    if returns.empty:
        return 0.0

    # Calculate deviations below the target return
    excess_ret = returns - target_return
    downside_diffs = np.minimum(0.0, excess_ret)
    
    # Quadratic mean of negative deviations
    downside_variance = np.sum(downside_diffs ** 2) / len(returns)
    downside_std = np.sqrt(downside_variance)
    
    return float(downside_std * np.sqrt(annualization_factor))

def calculate_drawdown_series(portfolio_value: pd.Series) -> pd.Series:
    """
    Computes daily drawdown series.

    Args:
        portfolio_value (pd.Series): Daily rupee value of the portfolio.

    Returns:
        pd.Series: Daily drawdown series (non-negative decimals, e.g. 0.05 for 5% drawdown).
    """
    peaks = portfolio_value.cummax()
    # Avoid division by zero if peak is 0
    drawdown = (peaks - portfolio_value) / peaks.replace(0, np.nan)
    return drawdown.fillna(0.0)

def calculate_maximum_drawdown(portfolio_value: pd.Series) -> Tuple[float, int, pd.Timestamp, pd.Timestamp]:
    """
    Calculates Maximum Drawdown (MDD), its duration, peak date, and recovery date.

    Args:
        portfolio_value (pd.Series): Daily portfolio value series.

    Returns:
        Tuple[float, int, pd.Timestamp, pd.Timestamp]:
            - max_dd (float): Maximum drawdown depth (as decimal, e.g., 0.15 for 15%).
            - duration (int): Peak-to-recovery duration in trading days.
            - peak_date (pd.Timestamp): Date when the peak occurred.
            - recovery_date (pd.Timestamp): Date when portfolio recovered to the previous peak.
    """
    if portfolio_value.empty:
        return 0.0, 0, pd.Timestamp("NaT"), pd.Timestamp("NaT")

    peaks = portfolio_value.cummax()
    drawdowns = (peaks - portfolio_value) / peaks.replace(0, np.nan)
    drawdowns = drawdowns.fillna(0.0)
    
    max_dd = float(drawdowns.max())
    if max_dd == 0.0:
        return 0.0, 0, portfolio_value.index[0], portfolio_value.index[0]

    trough_idx = drawdowns.idxmax()
    peak_date = peaks.loc[:trough_idx].idxmax()

    # Find recovery date (first date after trough where portfolio_value >= peak_value)
    peak_val = portfolio_value.loc[peak_date]
    post_trough = portfolio_value.loc[trough_idx:]
    recovery_series = post_trough[post_trough >= peak_val]

    if not recovery_series.empty:
        recovery_date = recovery_series.index[0]
        # Duration is from peak to recovery
        duration = int(len(portfolio_value.loc[peak_date:recovery_date]) - 1)
    else:
        recovery_date = pd.Timestamp("NaT")
        # If not recovered, duration is peak to end of dataset
        duration = int(len(portfolio_value.loc[peak_date:]) - 1)

    return max_dd, duration, peak_date, recovery_date

def calculate_ulcer_index(portfolio_value: pd.Series) -> float:
    """
    Calculates the Ulcer Index (UI) for the portfolio.

    The Ulcer Index measures the depth and duration of drawdowns.
    Formula:
        UI = sqrt( mean(Drawdown_t^2) )

    Args:
        portfolio_value (pd.Series): Daily portfolio value series.

    Returns:
        float: Ulcer Index value (as a ratio, e.g. 0.03 for 3%).
    """
    if portfolio_value.empty:
        return 0.0
    drawdowns = calculate_drawdown_series(portfolio_value)
    return float(np.sqrt(np.mean(drawdowns ** 2)))

def calculate_semi_deviation(returns: pd.Series) -> float:
    """
    Calculates the sample semi-deviation (standard deviation of returns below the mean).

    Args:
        returns (pd.Series): Series of strategy returns.

    Returns:
        float: Daily semi-deviation.
    """
    if returns.empty:
        return 0.0
    mean_ret = returns.mean()
    negative_diffs = returns[returns < mean_ret]
    if len(negative_diffs) <= 1:
        return 0.0
    return float(negative_diffs.std(ddof=1))

def calculate_var_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculates Historical Value at Risk (VaR) and Conditional Value at Risk (CVaR).

    Args:
        returns (pd.Series): Series of strategy returns.
        confidence_level (float): Confidence level for estimation (default is 0.95).

    Returns:
        Tuple[float, float]: (historical_var, cvar) as positive decimal fractions of capital.
    """
    if returns.empty:
        return 0.0, 0.0

    percentile = (1.0 - confidence_level) * 100.0
    var_val = -float(np.percentile(returns, percentile))

    # CVaR is the mean of all returns that fall below the negative VaR threshold
    tail_returns = returns[returns <= -var_val]
    if not tail_returns.empty:
        cvar_val = -float(tail_returns.mean())
    else:
        cvar_val = var_val

    return max(0.0, var_val), max(0.0, cvar_val)
