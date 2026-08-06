"""
Market Regime Detector Module.

Provides functions to identify market environments based on long-term trend,
realized volatility distributions, and crash/recovery states.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def detect_trend_regimes(
    prices: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    close_col: str = "Close"
) -> pd.Series:
    """
    Identifies market trend regimes based on moving averages.

    Rules:
        - Trending Up: Price > long_SMA AND short_SMA > long_SMA
        - Trending Down: Price < long_SMA AND short_SMA < long_SMA
        - Sideways: Otherwise

    Args:
        prices (pd.DataFrame): Price history DataFrame.
        short_window (int): Lookback for short SMA. Default is 50.
        long_window (int): Lookback for long SMA. Default is 200.
        close_col (str): Column name for close price. Default is 'Close'.

    Returns:
        pd.Series: Trend regime labels ('Trending_Up', 'Trending_Down', 'Sideways').
    """
    close = prices[close_col]
    
    # Calculate moving averages
    short_sma = close.rolling(short_window).mean()
    long_sma = close.rolling(long_window).mean()

    trend = pd.Series("Sideways", index=prices.index)

    # Apply masks
    up_mask = (close > long_sma) & (short_sma > long_sma)
    down_mask = (close < long_sma) & (short_sma < long_sma)

    trend[up_mask] = "Trending_Up"
    trend[down_mask] = "Trending_Down"

    # Handle warm-up lookback period as 'Sideways' or forward-filled
    warmup_mask = long_sma.isna()
    trend[warmup_mask] = "Sideways"

    logger.info(f"Trend regimes classified: Up={up_mask.sum()}, Down={down_mask.sum()}, Sideways={len(trend) - up_mask.sum() - down_mask.sum()}")
    return trend

def detect_volatility_regimes(
    prices: pd.DataFrame,
    window: int = 20,
    close_col: str = "Close"
) -> pd.Series:
    """
    Identifies volatility regimes using expanding quantiles of realized volatility.

    Realized volatility is computed as the rolling standard deviation of returns.
    Quantile thresholds (33% and 66%) are calculated expanding-window style
    to prevent look-ahead bias.

    Args:
        prices (pd.DataFrame): Price history.
        window (int): Lookback window for realized volatility. Default is 20.
        close_col (str): Column name for close price. Default is 'Close'.

    Returns:
        pd.Series: Volatility regime labels ('Low_Vol', 'Medium_Vol', 'High_Vol').
    """
    returns = prices[close_col].pct_change()
    
    # 20-day realized volatility annualized
    realized_vol = returns.rolling(window).std() * np.sqrt(252.0)
    realized_vol = realized_vol.bfill()  # Handle warm-up NaN

    # Expanding quantiles to avoid future look-ahead leaks
    q33 = realized_vol.expanding(min_periods=window).quantile(0.33)
    q66 = realized_vol.expanding(min_periods=window).quantile(0.66)
    
    # Backfill quantiles for the initial warm-up period
    q33 = q33.bfill()
    q66 = q66.bfill()

    vol_regime = pd.Series("Medium_Vol", index=prices.index)
    
    vol_regime[realized_vol <= q33] = "Low_Vol"
    vol_regime[realized_vol > q66] = "High_Vol"

    logger.info(f"Volatility regimes classified: Low_Vol={(realized_vol <= q33).sum()}, High_Vol={(realized_vol > q66).sum()}, Medium_Vol={len(vol_regime) - (realized_vol <= q33).sum() - (realized_vol > q66).sum()}")
    return vol_regime

def detect_combined_regimes(
    prices: pd.DataFrame,
    close_col: str = "Close"
) -> pd.DataFrame:
    """
    Combines trend and volatility regimes, and layers crash/recovery event states.

    Rules for overlays:
        - Crash: Drawdown from peak > 15% AND rolling 20-day return < -10%
        - Recovery: Drawdown from peak > 10% AND rolling 20-day return > 0.0

    Args:
        prices (pd.DataFrame): Price history.
        close_col (str): Column name for close price. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame containing individual and combined regime columns.
    """
    close = prices[close_col]
    
    trend = detect_trend_regimes(prices, close_col=close_col)
    vol = detect_volatility_regimes(prices, close_col=close_col)

    # Combined state string
    combined = trend + "_" + vol

    # Detect Crash / Recovery based on index drawdown & rolling return
    peaks = close.cummax()
    drawdowns = (peaks - close) / peaks
    returns_20d = close.pct_change(20)

    crash_mask = (drawdowns > 0.15) & (returns_20d < -0.10)
    recovery_mask = (drawdowns > 0.10) & (returns_20d > 0.0)

    # Apply overrides
    regime = combined.copy()
    regime[recovery_mask] = "Recovery"
    # Crash overrides recovery
    regime[crash_mask] = "Crash"

    # Realized volatility series for metrics
    realized_vol_20 = (close.pct_change().rolling(20).std() * np.sqrt(252.0)).bfill()
    realized_vol_50 = (close.pct_change().rolling(50).std() * np.sqrt(252.0)).bfill()
    realized_vol_100 = (close.pct_change().rolling(100).std() * np.sqrt(252.0)).bfill()

    regimes_df = pd.DataFrame({
        "Trend_Regime": trend,
        "Volatility_Regime": vol,
        "Combined_Regime": combined,
        "Final_Regime": regime,
        "Realized_Vol_20": realized_vol_20,
        "Realized_Vol_50": realized_vol_50,
        "Realized_Vol_100": realized_vol_100,
        "Drawdown": drawdowns
    }, index=prices.index)

    logger.info("Combined regime detection completed.")
    return regimes_df
