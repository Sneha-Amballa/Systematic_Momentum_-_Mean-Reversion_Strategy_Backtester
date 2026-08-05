import logging
import os
import pandas as pd
import yfinance as yf
from typing import List

class DataValidationError(Exception):
    """Custom exception raised when data validation fails."""
    pass

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens MultiIndex columns from yfinance to a single-level index,
    ensuring standard columns (Open, High, Low, Close, Volume) are accessible.
    
    Args:
        df (pd.DataFrame): Input DataFrame from yfinance.
        
    Returns:
        pd.DataFrame: DataFrame with single-level column headers.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    df = df.copy()
    target_cols = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
    
    # Identify which level of the MultiIndex contains our target columns
    found_level = -1
    for level in range(df.columns.nlevels):
        level_values = set(df.columns.get_level_values(level))
        if level_values.intersection(target_cols):
            found_level = level
            break
            
    if found_level != -1:
        df.columns = df.columns.get_level_values(found_level)
    else:
        # Fallback: Join levels with an underscore
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        
    return df

def download_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical daily market data for a given ticker from Yahoo Finance.

    Args:
        ticker (str): The ticker symbol (e.g., '^NSEI').
        start_date (str): Start date in 'YYYY-MM-DD' format (inclusive).
        end_date (str): End date in 'YYYY-MM-DD' format (exclusive).

    Returns:
        pd.DataFrame: Raw downloaded data.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading historical data for ticker '{ticker}' from {start_date} to {end_date}...")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        df = flatten_columns(df)
        return df
    except Exception as e:
        logger.error(f"Failed to download data for {ticker}: {str(e)}")
        raise RuntimeError(f"Data download failed due to network or API issue: {str(e)}") from e

def validate_data(df: pd.DataFrame) -> None:
    """
    Performs comprehensive validation checks on the downloaded daily data.

    Checks:
    1. DataFrame is not empty.
    2. Required columns exist: Open, High, Low, Close, Volume.
    3. Index is a DatetimeIndex.
    4. Dates are sorted in ascending order.
    5. Duplicate timestamps do not exist.
    6. Critical price columns (Open, High, Low, Close) contain positive values.
    7. Volume contains non-negative values.
    8. No missing (NaN) values exist in required columns.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Raises:
        DataValidationError: If any validation check fails.
    """
    logger = logging.getLogger(__name__)
    logger.info("Initiating data validation checks...")

    # 1. Empty Check
    if df.empty:
        raise DataValidationError("Validation failed: DataFrame is empty.")

    # 2. Required Columns Check
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Validation failed: Missing required columns: {missing_cols}")

    # 3. Datetime Index Check
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("Validation failed: Index is not a DatetimeIndex.")

    # 4. Chronological Sorting Check
    if not df.index.is_monotonic_increasing:
        raise DataValidationError("Validation failed: Index dates are not sorted in ascending order.")

    # 5. Duplicate Timestamps Check
    if df.index.duplicated().any():
        duplicate_dates = df.index[df.index.duplicated()].unique()
        raise DataValidationError(f"Validation failed: Duplicate dates found: {list(duplicate_dates)}")

    # 6. Missing Values Check
    for col in required_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            raise DataValidationError(f"Validation failed: Column '{col}' contains {missing_count} NaN values.")

    # 7. Price Positivity Check
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        non_positive_mask = df[col] <= 0
        if non_positive_mask.any():
            invalid_dates = df.index[non_positive_mask].strftime("%Y-%m-%d").tolist()
            raise DataValidationError(
                f"Validation failed: Column '{col}' contains non-positive values on dates: {invalid_dates[:5]}"
            )

    # 8. Volume Non-negativity Check
    negative_volume_mask = df["Volume"] < 0
    if negative_volume_mask.any():
        invalid_dates = df.index[negative_volume_mask].strftime("%Y-%m-%d").tolist()
        raise DataValidationError(
            f"Validation failed: Column 'Volume' contains negative values on dates: {invalid_dates[:5]}"
        )

    # Calculate statistics
    num_rows = len(df)
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    missing_values = df[list(required_cols)].isna().sum().to_dict()

    # Print / Log statistics
    logger.info("=== Data Statistics ===")
    logger.info(f"Number of rows: {num_rows}")
    logger.info(f"Start date: {start_date}")
    logger.info(f"End date: {end_date}")
    logger.info(f"Missing values per column: {missing_values}")
    logger.info("=== Data Validation Completed Successfully ===")

def save_data(df: pd.DataFrame, csv_path: str, parquet_path: str) -> None:
    """
    Saves the validated DataFrame to both CSV and Parquet formats.

    Args:
        df (pd.DataFrame): The validated DataFrame.
        csv_path (str): Destination path for the CSV output.
        parquet_path (str): Destination path for the Parquet output.
    """
    logger = logging.getLogger(__name__)
    logger.info("Saving validated dataset locally...")

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    try:
        # Save CSV
        df.to_csv(csv_path, index=True)
        logger.info(f"Successfully saved CSV dataset to: {csv_path}")

        # Save Parquet
        df.to_parquet(parquet_path, index=True, engine="pyarrow")
        logger.info(f"Successfully saved Parquet dataset to: {parquet_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise IOError(f"Failed to save datasets: {str(e)}") from e
