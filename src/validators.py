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

