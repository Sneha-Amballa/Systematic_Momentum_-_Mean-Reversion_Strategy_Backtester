"""
Performance Metrics and Performance Ratios Module.

Provides functions to compute return profiles, risk-adjusted metrics,
and statistical distribution metrics for trading strategies.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from src.risk import calculate_volatility, calculate_downside_volatility, calculate_maximum_drawdown

logger = logging.getLogger(__name__)

def calculate_return_metrics(
    portfolio_value: pd.Series,
    returns: pd.Series,
    annualization_factor: float = 252.0
) -> Dict[str, float]:
    """
    Computes key performance return metrics.

    Args:
        portfolio_value (pd.Series): Daily equity curve series.
        returns (pd.Series): Daily net returns series.
        annualization_factor (float): Daily periods in a year. Default is 252.0.

    Returns:
        Dict[str, float]: Return metrics dictionary.
    """
    if portfolio_value.empty or returns.empty:
        return {}

    n_days = len(returns)
    years = n_days / annualization_factor

    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1.0

    # CAGR: (V_end / V_start) ^ (1 / years) - 1
    if years > 0 and portfolio_value.iloc[0] > 0 and portfolio_value.iloc[-1] > 0:
        cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    avg_daily_return = returns.mean()
    avg_monthly_return = avg_daily_return * 21.0
    avg_annual_return = avg_daily_return * annualization_factor

    # Geometric Mean Return
    # Compounding returns must be positive
    pos_returns = 1.0 + returns
    if (pos_returns > 0).all():
        geometric_mean = float(np.exp(np.mean(np.log(pos_returns))) - 1.0)
    else:
        geometric_mean = 0.0

    arithmetic_mean = float(avg_daily_return)

    return {
        "Total_Return": float(total_return),
        "CAGR": float(cagr),
        "Average_Daily_Return": float(avg_daily_return),
        "Average_Monthly_Return": float(avg_monthly_return),
        "Average_Annual_Return": float(avg_annual_return),
        "Arithmetic_Mean_Return": arithmetic_mean,
        "Geometric_Mean_Return": geometric_mean,
        "Cumulative_Return": float(total_return)
    }

def calculate_risk_adjusted_ratios(
    returns: pd.Series,
    portfolio_value: pd.Series,
    benchmark_returns: pd.Series = None,
    beta: float = None,
    risk_free_rate_annual: float = 0.0,
    annualization_factor: float = 252.0
) -> Dict[str, float]:
    """
    Calculates key risk-adjusted performance ratios.

    Ratios calculated:
    - Sharpe Ratio (annualized)
    - Sortino Ratio (annualized)
    - Calmar Ratio (annualized)
    - Information Ratio (annualized, relative to benchmark if provided)
    - Treynor Ratio (annualized, using beta if provided)
    - Omega Ratio (daily threshold = risk-free rate)

    Args:
        returns (pd.Series): Daily net returns of strategy.
        portfolio_value (pd.Series): Daily capital curve.
        benchmark_returns (pd.Series): Daily returns of benchmark.
        beta (float): Strategy Beta relative to benchmark.
        risk_free_rate_annual (float): Annual risk-free rate. Default is 0.0.
        annualization_factor (float): Annual periods. Default is 252.0.

    Returns:
        Dict[str, float]: Risk-adjusted ratios.
    """
    daily_rf = risk_free_rate_annual / annualization_factor
    excess_returns = returns - daily_rf

    # 1. Sharpe Ratio
    vol = calculate_volatility(returns, annualization_factor)
    sharpe = (excess_returns.mean() / returns.std(ddof=1)) * np.sqrt(annualization_factor) if vol > 0 else 0.0

    # 2. Sortino Ratio
    downside_vol = calculate_downside_volatility(returns, daily_rf, annualization_factor)
    sortino = (excess_returns.mean() / (downside_vol / np.sqrt(annualization_factor))) * np.sqrt(annualization_factor) if downside_vol > 0 else 0.0

    # 3. Calmar Ratio
    cagr = calculate_return_metrics(portfolio_value, returns, annualization_factor).get("CAGR", 0.0)
    max_dd, _, _, _ = calculate_maximum_drawdown(portfolio_value)
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # 4. Information Ratio
    info_ratio = 0.0
    if benchmark_returns is not None:
        tracking_difference = returns - benchmark_returns
        tracking_error = tracking_difference.std(ddof=1) * np.sqrt(annualization_factor)
        if tracking_error > 0:
            info_ratio = (tracking_difference.mean() * annualization_factor) / tracking_error

    # 5. Treynor Ratio
    treynor = 0.0
    if beta is not None and beta != 0.0:
        annualized_excess_return = returns.mean() * annualization_factor - risk_free_rate_annual
        treynor = annualized_excess_return / beta

    # 6. Omega Ratio (daily gain sum / daily loss sum at threshold daily_rf)
    gains = excess_returns[excess_returns > 0].sum()
    losses = -excess_returns[excess_returns < 0].sum()
    omega = gains / losses if losses > 0 else np.nan

    return {
        "Sharpe_Ratio": float(sharpe),
        "Sortino_Ratio": float(sortino),
        "Calmar_Ratio": float(calmar),
        "Information_Ratio": float(info_ratio),
        "Treynor_Ratio": float(treynor),
        "Omega_Ratio": float(omega)
    }

def calculate_distribution_metrics(returns: pd.Series) -> Dict[str, Any]:
    """
    Computes returns distribution analysis parameters.

    Args:
        returns (pd.Series): Daily net returns.

    Returns:
        Dict[str, Any]: Distribution metrics and test statistics.
    """
    if returns.empty:
        return {}

    mean_ret = float(returns.mean())
    median_ret = float(returns.median())
    variance_ret = float(returns.var(ddof=1))
    std_ret = float(returns.std(ddof=1))
    
    skew = float(returns.skew())
    kurt = float(returns.kurt())  # excess kurtosis

    # Jarque-Bera Test
    # Standard formula: JB = (N/6) * (S^2 + (K - 3)^2 / 4)
    # scipy jarque_bera returns (jb_stat, p_value)
    if len(returns) >= 2:
        jb_stat, p_val = stats.jarque_bera(returns)
    else:
        jb_stat, p_val = 0.0, 1.0

    # Percentiles
    percentiles = {
        "P1": float(np.percentile(returns, 1)),
        "P5": float(np.percentile(returns, 5)),
        "P10": float(np.percentile(returns, 10)),
        "P25": float(np.percentile(returns, 25)),
        "P50": float(np.percentile(returns, 50)),
        "P75": float(np.percentile(returns, 75)),
        "P90": float(np.percentile(returns, 90)),
        "P95": float(np.percentile(returns, 95)),
        "P99": float(np.percentile(returns, 99))
    }

    return {
        "Mean": mean_ret,
        "Median": median_ret,
        "Variance": variance_ret,
        "Standard_Deviation": std_ret,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Jarque_Bera_Stat": float(jb_stat),
        "Jarque_Bera_PValue": float(p_val),
        "Percentiles": percentiles
    }
