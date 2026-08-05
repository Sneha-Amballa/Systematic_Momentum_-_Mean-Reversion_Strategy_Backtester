"""
Technical Indicator Construction Module.

Provides pure mathematical indicator functions for quantitative analysis.
These functions operate on pandas Series and are decoupled from data loading or strategy logic.
"""

import logging
import pandas as pd

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
