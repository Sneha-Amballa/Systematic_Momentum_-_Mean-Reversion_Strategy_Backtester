import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BacktesterException(Exception):
    """Base exception for the Systematic Backtester project."""
    pass

class StructuralValidationError(BacktesterException):
    """Exception raised when structural data constraints are violated."""
    pass

class DataQualityValidationError(BacktesterException):
    """Exception raised when data quality validation checks fail."""
    pass

def validate_structure(df: pd.DataFrame) -> None:
    """
    Validates the structure of the input DataFrame.

    Checks:
    - The DataFrame is not empty.
    - The index is a DatetimeIndex.
    - The index is sorted chronologically.
    - The index is unique (no duplicate timestamps).
    - Required OHLCV columns exist.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Raises:
        StructuralValidationError: If any structural validation check fails.
    """
    # 1. Empty Check
    if df.empty:
        raise StructuralValidationError("DataFrame is empty.")

    # 2. DatetimeIndex Check
    if not isinstance(df.index, pd.DatetimeIndex):
        raise StructuralValidationError(
            f"Expected DatetimeIndex, but got index of type '{type(df.index).__name__}'."
        )

    # 3. Sorted Index Check
    if not df.index.is_monotonic_increasing:
        raise StructuralValidationError("DatetimeIndex is not sorted in ascending chronological order.")

    # 4. Unique Index Check
    if df.index.duplicated().any():
        num_duplicates = df.index.duplicated().sum()
        raise StructuralValidationError(f"DatetimeIndex contains {num_duplicates} duplicate timestamps.")

    # 5. Required Columns Check
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise StructuralValidationError(f"Missing required columns: {missing_columns}")

def run_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Scans the DataFrame for data quality anomalies and returns a report.
    Logs every anomaly found at the WARNING level.

    Scans for:
    - Fully duplicate rows
    - Duplicate timestamps
    - Missing (NaN/Null) values
    - Infinite values
    - Zero or negative prices (OHLC)
    - Illogical price relationships (High < Low, High < Open, High < Close, Low > Open, Low > Close)

    Args:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        Dict[str, Any]: A quality report dictionary detailing the issues found.
    """
    report: Dict[str, Any] = {}
    price_cols = ["Open", "High", "Low", "Close"]
    
    # 1. Duplicate rows check
    dup_rows = int(df.duplicated().sum())
    report["duplicate_rows"] = dup_rows
    if dup_rows > 0:
        logger.warning(f"Found {dup_rows} fully duplicate rows in the dataset.")

    # 2. Duplicate timestamps check
    dup_timestamps = int(df.index.duplicated().sum())
    report["duplicate_timestamps"] = dup_timestamps
    if dup_timestamps > 0:
        logger.warning(f"Found {dup_timestamps} duplicate timestamps in the index.")

    # 3. Missing values check
    missing_values = df.isna().sum().to_dict()
    report["missing_values"] = missing_values
    for col, count in missing_values.items():
        if count > 0:
            logger.warning(f"Column '{col}' has {count} missing (NaN) values.")

    # 4. Infinite values check
    numeric_df = df.select_dtypes(include=[np.number])
    inf_values = np.isinf(numeric_df).sum().to_dict()
    report["infinite_values"] = inf_values
    for col, count in inf_values.items():
        if count > 0:
            logger.warning(f"Column '{col}' has {count} infinite values.")

    # 5. Negative or Zero prices
    non_positive_prices = {}
    for col in price_cols:
        invalid_mask = df[col] <= 0
        invalid_count = int(invalid_mask.sum())
        non_positive_prices[col] = invalid_count
        if invalid_count > 0:
            invalid_dates = df.index[invalid_mask].strftime("%Y-%m-%d").tolist()
            logger.warning(
                f"Column '{col}' contains {invalid_count} zero or negative values. "
                f"Dates (first 5): {invalid_dates[:5]}"
            )
    report["non_positive_prices"] = non_positive_prices

    # 6. Logical price checks
    logical_errors = {
        "High_lt_Low": int((df["High"] < df["Low"]).sum()),
        "High_lt_Open": int((df["High"] < df["Open"]).sum()),
        "High_lt_Close": int((df["High"] < df["Close"]).sum()),
        "Low_gt_Open": int((df["Low"] > df["Open"]).sum()),
        "Low_gt_Close": int((df["Low"] > df["Close"]).sum()),
    }
    report["logical_price_errors"] = logical_errors

    for error_name, count in logical_errors.items():
        if count > 0:
            # Determine logic violation details
            if error_name == "High_lt_Low":
                invalid_dates = df.index[df["High"] < df["Low"]].strftime("%Y-%m-%d").tolist()
                logger.warning(f"High price < Low price in {count} rows. Dates (first 5): {invalid_dates[:5]}")
            elif error_name == "High_lt_Open":
                invalid_dates = df.index[df["High"] < df["Open"]].strftime("%Y-%m-%d").tolist()
                logger.warning(f"High price < Open price in {count} rows. Dates (first 5): {invalid_dates[:5]}")
            elif error_name == "High_lt_Close":
                invalid_dates = df.index[df["High"] < df["Close"]].strftime("%Y-%m-%d").tolist()
                logger.warning(f"High price < Close price in {count} rows. Dates (first 5): {invalid_dates[:5]}")
            elif error_name == "Low_gt_Open":
                invalid_dates = df.index[df["Low"] > df["Open"]].strftime("%Y-%m-%d").tolist()
                logger.warning(f"Low price > Open price in {count} rows. Dates (first 5): {invalid_dates[:5]}")
            elif error_name == "Low_gt_Close":
                invalid_dates = df.index[df["Low"] > df["Close"]].strftime("%Y-%m-%d").tolist()
                logger.warning(f"Low price > Close price in {count} rows. Dates (first 5): {invalid_dates[:5]}")

    return report

def validate_clean_data(df: pd.DataFrame, keep_volume: bool) -> None:
    """
    Performs final structural and quality checks on the preprocessed clean dataset.

    Checks:
    - Index has no duplicates and is monotonically increasing.
    - No missing (NaN) values in OHLC columns.
    - Feature columns exist and have appropriate lengths.
    - All columns are of correct numeric (float64) types, with no object columns.

    Args:
        df (pd.DataFrame): The preprocessed clean DataFrame.
        keep_volume (bool): Whether the volume column is retained.

    Raises:
        DataQualityValidationError: If clean data validation checks fail.
    """
    # 1. Index integrity
    if df.empty:
        raise DataQualityValidationError("Final validation failed: Clean DataFrame is empty.")
    if not df.index.is_monotonic_increasing:
        raise DataQualityValidationError("Final validation failed: Clean index is not sorted chronologically.")
    if df.index.duplicated().any():
        raise DataQualityValidationError("Final validation failed: Clean index contains duplicates.")

    # 2. Key columns existence and check for NaNs
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if col not in df.columns:
            raise DataQualityValidationError(f"Final validation failed: Missing required price column '{col}'.")
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise DataQualityValidationError(f"Final validation failed: Column '{col}' contains {nan_count} missing values.")

    # 3. Volume existence check if keep_volume is True
    if keep_volume:
        if "Volume" not in df.columns:
            raise DataQualityValidationError("Final validation failed: 'Volume' column was expected but is missing.")
        # Volume could contain NaN values if they were present in raw, so we check and ensure no NaNs in Volume
        nan_volume = df["Volume"].isna().sum()
        if nan_volume > 0:
            raise DataQualityValidationError(f"Final validation failed: Column 'Volume' contains {nan_volume} missing values.")
    else:
        if "Volume" in df.columns:
            raise DataQualityValidationError("Final validation failed: 'Volume' column was dropped but is still present.")

    # 4. Engineered features existence check
    expected_features = ["Simple_Return", "Log_Return", "Price_Change", "Daily_Range", "Pct_Range"]
    for feature in expected_features:
        if feature not in df.columns:
            raise DataQualityValidationError(f"Final validation failed: Feature column '{feature}' was not created.")
        
        # Note: Return columns (Simple_Return, Log_Return) will have exactly 1 NaN on the very first row.
        # This is expected and checked:
        nan_count = df[feature].isna().sum()
        expected_nan = 1 if feature in ["Simple_Return", "Log_Return"] else 0
        if nan_count > expected_nan:
            raise DataQualityValidationError(
                f"Final validation failed: Feature '{feature}' has {nan_count} NaNs (expected <= {expected_nan})."
            )

    # 5. Data type checks
    for col in df.columns:
        if df[col].dtype == "object":
            raise DataQualityValidationError(f"Final validation failed: Column '{col}' is stored as object dtype.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataQualityValidationError(f"Final validation failed: Column '{col}' is not a numeric dtype.")

def validate_indicator_params(df: pd.DataFrame, short_window: int, long_window: int, close_col: str = "Close") -> None:
    """
    Validates input parameters for the moving average crossover indicators.

    Checks:
    - Short and long windows are positive integers.
    - Short window is strictly less than the long window.
    - Close column exists in the DataFrame.
    - The DataFrame contains enough historical observations to calculate the long SMA.

    Args:
        df (pd.DataFrame): Input market DataFrame.
        short_window (int): Lookback window for the short moving average.
        long_window (int): Lookback window for the long moving average.
        close_col (str): Name of the column containing closing prices.

    Raises:
        ValueError: If any parameters are mathematically or logically invalid.
        StructuralValidationError: If the required close column is missing or data is insufficient.
    """
    if not isinstance(short_window, int) or not isinstance(long_window, int):
        raise ValueError("Moving average window sizes must be integers.")
        
    if short_window <= 0:
        raise ValueError(f"Short window must be positive. Got: {short_window}")
        
    if long_window <= 0:
        raise ValueError(f"Long window must be positive. Got: {long_window}")
        
    if short_window >= long_window:
        raise ValueError(
            f"Short window ({short_window}) must be strictly less than long window ({long_window})."
        )
        
    if close_col not in df.columns:
        raise StructuralValidationError(f"Required closing price column '{close_col}' not found in DataFrame.")
        
    if len(df) < long_window:
        raise StructuralValidationError(
            f"Insufficient data. Dataset has {len(df)} rows, but the strategy requires at least "
            f"{long_window} rows for the long moving average window."
        )

def validate_signals(df: pd.DataFrame, raw_col: str = "Raw_Signal", exec_col: str = "Execution_Signal", pos_col: str = "Position") -> None:
    """
    Validates that the generated strategy signals and positions are correct and bias-free.

    Checks:
    - Signal columns exist in the DataFrame.
    - Signal and position values contain only 0 and 1 (or NaN during the warm-up period).
    - Shift constraint: Execution_Signal[t] == Raw_Signal[t-1] for all t > 0.
    - No future leakage: Execution_Signal is shifted forward by 1, meaning it does not incorporate today's Raw_Signal.
    - No missing values in signal columns after the warm-up period.

    Args:
        df (pd.DataFrame): DataFrame containing closing prices and generated signals.
        raw_col (str): Column name for the raw trading signals.
        exec_col (str): Column name for the shifted execution signals.
        pos_col (str): Column name for the position indicators.

    Raises:
        DataQualityValidationError: If any signal verification or look-ahead check fails.
    """
    for col in [raw_col, exec_col, pos_col]:
        if col not in df.columns:
            raise DataQualityValidationError(f"Required signal column '{col}' is missing from DataFrame.")

    raw_series = df[raw_col]
    exec_series = df[exec_col]
    pos_series = df[pos_col]

    # 1. Check signal values are binary (0, 1) or NaN
    for name, series in [(raw_col, raw_series), (exec_col, exec_series), (pos_col, pos_series)]:
        unique_vals = series.dropna().unique()
        invalid_vals = [v for v in unique_vals if v not in [0, 1, 0.0, 1.0]]
        if invalid_vals:
            raise DataQualityValidationError(
                f"Column '{name}' contains invalid non-binary signal values: {invalid_vals}. "
                "Only 0 and 1 are allowed."
            )

    # 2. Verify look-ahead shift: Execution_Signal must equal Raw_Signal.shift(1)
    expected_exec = raw_series.shift(1)
    
    # We compare indices where expected_exec is not null
    compare_mask = expected_exec.notna()
    if not (exec_series[compare_mask] == expected_exec[compare_mask]).all():
        mismatches = df[exec_series != expected_exec]
        raise DataQualityValidationError(
            f"Look-ahead bias check failed. '{exec_col}' does not match '{raw_col}' shifted by 1. "
            f"Number of mismatches: {len(mismatches)}. First few mismatch dates: {mismatches.index[exec_series != expected_exec][:5].tolist()}"
        )

    # 3. Position must be equal to execution signal (or direct mapping)
    if not (pos_series[compare_mask] == exec_series[compare_mask]).all():
        mismatches = df[pos_series != exec_series]
        raise DataQualityValidationError(
            f"Position column mismatch. '{pos_col}' must be equal to '{exec_col}' where defined. "
            f"First few mismatch dates: {mismatches.index[pos_series != exec_series][:5].tolist()}"
        )

    # 4. Check for NaNs after the warm-up period
    first_valid_idx = raw_series.first_valid_index()
    if first_valid_idx is not None:
        first_valid_loc = df.index.get_loc(first_valid_idx)
        
        nan_count_raw = raw_series.iloc[first_valid_loc:].isna().sum()
        if nan_count_raw > 0:
            raise DataQualityValidationError(
                f"Column '{raw_col}' contains {nan_count_raw} missing values after the strategy warm-up period."
            )

        exec_start_loc = first_valid_loc + 1
        if exec_start_loc < len(df):
            nan_count_exec = exec_series.iloc[exec_start_loc:].isna().sum()
            if nan_count_exec > 0:
                raise DataQualityValidationError(
                    f"Column '{exec_col}' contains {nan_count_exec} missing values after the execution start period."
                )
            nan_count_pos = pos_series.iloc[exec_start_loc:].isna().sum()
            if nan_count_pos > 0:
                raise DataQualityValidationError(
                    f"Column '{pos_col}' contains {nan_count_pos} missing values after the execution start period."
                )

def validate_mean_reversion_params(
    df: pd.DataFrame, 
    window: int, 
    entry_threshold: float, 
    exit_threshold: float, 
    close_col: str = "Close"
) -> None:
    """
    Validates input parameters for the rolling mean reversion strategy.

    Checks:
    - Window is an integer and window > 1.
    - Window is strictly smaller than the dataset length.
    - Close column exists in the DataFrame.
    - Entry threshold is strictly less than the exit threshold.

    Args:
        df (pd.DataFrame): Input market DataFrame.
        window (int): Rolling lookback window size.
        entry_threshold (float): Z-score threshold for entry (e.g. -2.0).
        exit_threshold (float): Z-score threshold for exit (e.g. -0.5).
        close_col (str): Column name containing closing prices.

    Raises:
        ValueError: If any parameters are mathematically or logically invalid.
        StructuralValidationError: If the required close column is missing or data is insufficient.
    """
    if not isinstance(window, int):
        raise ValueError("Rolling window size must be an integer.")

    if window <= 1:
        raise ValueError(f"Rolling window size must be greater than 1. Got: {window}")

    if close_col not in df.columns:
        raise StructuralValidationError(f"Required closing price column '{close_col}' not found in DataFrame.")

    if len(df) < window:
        raise StructuralValidationError(
            f"Insufficient data. Dataset has {len(df)} rows, but the strategy requires at least "
            f"{window} rows for the rolling window."
        )

    if entry_threshold >= exit_threshold:
        raise ValueError(
            f"Entry threshold ({entry_threshold}) must be strictly less than exit threshold ({exit_threshold}) "
            f"for a long-only mean reversion strategy."
        )

def validate_backtest_inputs(prices: pd.DataFrame, signals: pd.Series) -> None:
    """
    Validates price data and signal data inputs before running the backtester.

    Checks:
    - Inputs are correct types (DataFrame and Series).
    - Both have DatetimeIndex.
    - Index alignments match.
    - Prices contain Close prices.
    - Signals contain only binary values (0, 1) or NaN.

    Args:
        prices (pd.DataFrame): Market price history.
        signals (pd.Series): Strategy signal series.

    Raises:
        TypeError: If inputs are of wrong types.
        StructuralValidationError: If indices are misaligned or required columns are missing.
        DataQualityValidationError: If signal constraints are violated.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if not isinstance(signals, pd.Series):
        raise TypeError("signals must be a pandas Series.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise StructuralValidationError("prices index must be a DatetimeIndex.")
    if not isinstance(signals.index, pd.DatetimeIndex):
        raise StructuralValidationError("signals index must be a DatetimeIndex.")

    if not prices.index.is_monotonic_increasing:
        raise StructuralValidationError("prices index must be chronologically sorted.")
    if not signals.index.is_monotonic_increasing:
        raise StructuralValidationError("signals index must be chronologically sorted.")

    if "Close" not in prices.columns:
        raise StructuralValidationError("prices must contain a 'Close' column.")

    # Check alignment
    missing_dates = signals.index.difference(prices.index)
    if len(missing_dates) > 0:
        raise StructuralValidationError(
            f"Signals index contains {len(missing_dates)} dates not present in prices. "
            f"First few missing dates: {missing_dates[:5].tolist()}"
        )

    # Check signal values (must be binary 0, 1 or NaN)
    unique_vals = signals.dropna().unique()
    invalid_vals = [v for v in unique_vals if v not in [0, 1, 0.0, 1.0, -1, -1.0]]
    if invalid_vals:
        raise DataQualityValidationError(
            f"Signals contain invalid non-binary values: {invalid_vals}. "
            "Only 0, 1, and -1 are allowed."
        )

def validate_backtest_results(df: pd.DataFrame, initial_capital: float) -> None:
    """
    Validates the backtest output DataFrame.

    Checks:
    - Required columns exist: Portfolio_Value, Daily_Return, Cumulative_Return, Daily_PnL.
    - No missing values in key columns.
    - Equity curve starts exactly at initial_capital.
    - Portfolio values are strictly positive (no bankruptcy).

    Args:
        df (pd.DataFrame): Simulated portfolio history.
        initial_capital (float): Configured starting capital.

    Raises:
        DataQualityValidationError: If result validation fails.
    """
    required_cols = ["Portfolio_Value", "Daily_Return", "Cumulative_Return", "Daily_PnL", "Equity_Curve"]
    for col in required_cols:
        if col not in df.columns:
            raise DataQualityValidationError(f"Backtest results missing required column: '{col}'")
        
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise DataQualityValidationError(f"Backtest results column '{col}' contains {nan_count} NaN values.")

    # Bankruptcy check
    min_val = df["Portfolio_Value"].min()
    if min_val <= 0:
        raise DataQualityValidationError(
            f"Portfolio bankruptcy detected. Minimum portfolio value: {min_val:.2f} <= 0."
        )

    # Initial capital check
    # The portfolio value must be exactly initial_capital on the first day if return is 0
    # or equal to initial_capital * (1 + Daily_Return)
    first_val = df["Portfolio_Value"].iloc[0]
    first_ret = df["Daily_Return"].iloc[0]
    expected_val = initial_capital * (1.0 + first_ret)
    if not np.isclose(first_val, expected_val, rtol=1e-5):
        raise DataQualityValidationError(
            f"First day Portfolio_Value ({first_val:.2f}) does not match "
            f"expected value ({expected_val:.2f}) based on return {first_ret:.6f}."
        )

def validate_analytics_inputs(portfolio_df: pd.DataFrame, trade_book: pd.DataFrame) -> None:
    """
    Validates strategy simulation outputs before running the metrics/analytics framework.

    Checks:
    - Inputs are correct types (DataFrames).
    - portfolio_df has DatetimeIndex.
    - Required columns exist in portfolio_df: Portfolio_Value, Daily_Return.
    - If trade_book is not empty, checks it contains required columns.
    - No NaN values in portfolio_df key columns.

    Args:
        portfolio_df (pd.DataFrame): Daily portfolio value and returns history.
        trade_book (pd.DataFrame): Trade book log.

    Raises:
        TypeError: If inputs are of incorrect types.
        StructuralValidationError: If indices are invalid or required columns are missing.
        DataQualityValidationError: If NaNs or data inconsistencies are detected.
    """
    if not isinstance(portfolio_df, pd.DataFrame):
        raise TypeError("portfolio_df must be a pandas DataFrame.")
    if not isinstance(trade_book, pd.DataFrame):
        raise TypeError("trade_book must be a pandas DataFrame.")

    if not isinstance(portfolio_df.index, pd.DatetimeIndex):
        raise StructuralValidationError("portfolio_df index must be a DatetimeIndex.")

    # Check portfolio columns
    required_port_cols = ["Portfolio_Value", "Daily_Return"]
    for col in required_port_cols:
        if col not in portfolio_df.columns:
            raise StructuralValidationError(f"portfolio_df is missing required column: '{col}'")
        nan_count = portfolio_df[col].isna().sum()
        if nan_count > 0:
            raise DataQualityValidationError(f"portfolio_df column '{col}' contains {nan_count} missing values.")

    # Check trade book columns if not empty
    if not trade_book.empty:
        required_trade_cols = [
            "Entry Date", "Exit Date", "Holding Days", 
            "Entry Price", "Exit Price", "Position Size", "Total Cost"
        ]
        for col in required_trade_cols:
            if col not in trade_book.columns:
                raise StructuralValidationError(f"trade_book is missing required column: '{col}'")




