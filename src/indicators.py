"""
Technical Indicator Construction Module.

Provides pure mathematical indicator functions for quantitative analysis.
These functions operate on pandas Series and are decoupled from data loading or strategy logic.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA) for a given pandas Series.

    The first `window - 1` observations in the returned series will contain NaN
    due to insufficient historical lookback data to compute a complete mean.

    Args:
        series (pd.Series): A pandas Series of numeric values (typically Close prices).
        window (int): The moving average lookback window size (must be >= 1).

    Returns:
        pd.Series: A series containing the calculated SMA values, named 'SMA_<window>'.

    Raises:
        ValueError: If the window size is less than 1, or if the series is not numeric.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input series must be a pandas Series object.")
    if window < 1:
        raise ValueError(f"Window size must be a positive integer >= 1. Got: {window}")
    
    logger.debug(f"Calculating SMA with window={window} on series '{series.name or 'input'}'")
    sma_series = series.rolling(window=window).mean()
    sma_series.name = f"SMA_{window}"
    return sma_series

def calculate_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Calculates the rolling mean for a given pandas Series.

    Args:
        series (pd.Series): A pandas Series of numeric values (typically Close prices).
        window (int): The rolling window size (must be >= 1).

    Returns:
        pd.Series: A series containing the calculated rolling mean values, named 'Rolling_Mean_<window>'.

    Raises:
        ValueError: If the window size is less than 1, or if the series is not numeric.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input series must be a pandas Series object.")
    if window < 1:
        raise ValueError(f"Window size must be a positive integer >= 1. Got: {window}")

    logger.debug(f"Calculating rolling mean with window={window} on series '{series.name or 'input'}'")
    mean_series = series.rolling(window=window).mean()
    mean_series.name = f"Rolling_Mean_{window}"
    return mean_series

def calculate_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """
    Calculates the rolling standard deviation for a given pandas Series.

    Args:
        series (pd.Series): A pandas Series of numeric values (typically Close prices).
        window (int): The rolling window size (must be >= 2).

    Returns:
        pd.Series: A series containing the calculated rolling standard deviation values, named 'Rolling_STD_<window>'.

    Raises:
        ValueError: If the window size is less than 2, or if the series is not numeric.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input series must be a pandas Series object.")
    if window < 2:
        raise ValueError(f"Window size for standard deviation must be a positive integer >= 2. Got: {window}")

    logger.debug(f"Calculating rolling standard deviation with window={window} on series '{series.name or 'input'}'")
    std_series = series.rolling(window=window).std()
    std_series.name = f"Rolling_STD_{window}"
    return std_series

def calculate_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Calculates the rolling Z-Score for a given pandas Series relative to its rolling mean and std.

    Formula:
        Z = (Close - Rolling_Mean) / Rolling_STD

    Safe division is implemented to handle cases where Rolling_STD is zero or close to it.
    Zero standard deviations are replaced with NaN in the output Z-score.

    Args:
        series (pd.Series): A pandas Series of numeric values (typically Close prices).
        window (int): The rolling window size (must be >= 2).

    Returns:
        pd.Series: A series containing the calculated rolling Z-score values, named 'Z_Score_<window>'.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input series must be a pandas Series object.")
    
    logger.info(f"Computing rolling Z-Score with window={window} on series '{series.name or 'input'}'")
    
    # Calculate rolling metrics
    rolling_mean = calculate_rolling_mean(series, window)
    rolling_std = calculate_rolling_std(series, window)
    
    # Handle division by zero / std is zero safely
    # If rolling_std is 0 or NaN, Z-Score should be NaN
    # We can use pd.Series math and then clean up any infinities.
    zscore = (series - rolling_mean) / rolling_std
    
    # Replace infinite values with NaN
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    zscore.name = f"Z_Score_{window}"
    
    return zscore
