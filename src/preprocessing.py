import logging
import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.validators import validate_structure, run_quality_checks, validate_clean_data

logger = logging.getLogger(__name__)

class PreprocessingError(Exception):
    """Exception raised when preprocessing steps fail."""
    pass

class DataPreprocessor:
    """
    Orchestrates the 11-stage preprocessing pipeline to transform raw market data
    into a clean, validated, analysis-ready dataset.
    """

    def __init__(self, keep_volume: bool = False, outlier_threshold: float = 0.08):
        """
        Initializes the DataPreprocessor with configurations.

        Args:
            keep_volume (bool): If True, retains the Volume column. If False (default), drops it.
            outlier_threshold (float): Return threshold to flag outliers (default: 0.08).
        """
        self.keep_volume = keep_volume
        self.outlier_threshold = outlier_threshold
        self.summary_report: Dict[str, Any] = {}

    def load_raw_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Stage 1: Loads the raw parquet dataset and logs metadata.

        Args:
            file_path (str): Path to the raw parquet file.

        Returns:
            pd.DataFrame: Loaded raw market data.
        """
        logger.info(f"Stage 1: Loading raw dataset from '{file_path}'...")
        if not os.path.exists(file_path):
            raise PreprocessingError(f"Raw data file not found at: {file_path}")

        try:
            df = pd.read_parquet(file_path)
            
            # Baseline metrics
            num_rows, num_cols = df.shape
            start_date = df.index.min().strftime("%Y-%m-%d") if not df.empty else "N/A"
            end_date = df.index.max().strftime("%Y-%m-%d") if not df.empty else "N/A"
            
            logger.info(f"Raw data loaded successfully: {num_rows} rows, {num_cols} columns.")
            logger.info(f"Raw Date Range: {start_date} to {end_date}")
            
            self.summary_report["initial_rows"] = num_rows
            self.summary_report["initial_cols"] = num_cols
            self.summary_report["initial_date_range"] = (start_date, end_date)
            
            return df
        except Exception as e:
            logger.error(f"Failed to load raw parquet data: {str(e)}")
            raise PreprocessingError(f"Error loading raw data: {str(e)}") from e

    def process(self, raw_path: str, clean_parquet_path: str, clean_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the full 11-stage preprocessing pipeline.

        Args:
            raw_path (str): Filepath of the raw input parquet file.
            clean_parquet_path (str): Output filepath for clean parquet.
            clean_csv_path (str): Output filepath for clean CSV.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - The fully cleaned and feature-engineered DataFrame.
                - A DataFrame containing flagged outlier observations.
        """
        logger.info("Starting Data Preprocessing Pipeline...")

        # Stage 1: Load
        df = self.load_raw_dataset(raw_path)

        # Stage 2: Structural Validation
        logger.info("Stage 2: Running structural validation checks...")
        validate_structure(df)
        logger.info("Structural validation checks passed.")

        # Stage 3: Data Quality Checks
        logger.info("Stage 3: Running data quality anomaly scan...")
        quality_report = run_quality_checks(df)
        self.summary_report["duplicate_rows_found"] = quality_report["duplicate_rows"]
        self.summary_report["duplicate_timestamps_found"] = quality_report["duplicate_timestamps"]
        self.summary_report["missing_values_found"] = sum(quality_report["missing_values"].values())
        logger.info("Data quality anomaly scan completed.")

        # Stage 4: Handle Missing Data
        logger.info("Stage 4: Handling missing price data...")
        # Get count of initial rows before dropping
        rows_before_drop = len(df)
        
        # We explicitly drop rows where any of the core OHLC price columns are NaN.
        # We do NOT drop rows based on missing Volume, as volume is processed separately.
        price_cols = ["Open", "High", "Low", "Close"]
        df_cleaned = df.dropna(subset=price_cols).copy()
        
        # Also drop fully duplicated rows if any exist
        df_cleaned = df_cleaned.drop_duplicates()
        
        rows_removed = rows_before_drop - len(df_cleaned)
        logger.info(
            f"Dropped {rows_removed} rows containing missing price values or duplicate entries. "
            "Note: Reindexing to calendar days and forward-filling price values are intentionally avoided "
            "to prevent fabricating non-existent trading days, preserving natural market closures and weekends."
        )
        self.summary_report["rows_removed_missing_or_duplicate"] = rows_removed

        # Stage 5: Remove Unnecessary Columns (Volume)
        logger.info(f"Stage 5: Column filter step (keep_volume={self.keep_volume})...")
        cols_dropped = []
        if not self.keep_volume:
            if "Volume" in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=["Volume"])
                cols_dropped.append("Volume")
                logger.info("Dropped 'Volume' column as it is configured to be dropped.")
        else:
            logger.info("Retained 'Volume' column as configured.")
        self.summary_report["columns_dropped"] = cols_dropped

        # Stage 6: Data Type Optimization
        logger.info("Stage 6: Optimizing numeric data types...")
        mem_before = df_cleaned.memory_usage(deep=True).sum()
        
        # Ensure all price columns are strict float64
        for col in price_cols:
            df_cleaned[col] = df_cleaned[col].astype("float64")
            
        if self.keep_volume and "Volume" in df_cleaned.columns:
            # Volume can be cast to float64 or int64; we use float64 to handle any potential NaNs safely
            df_cleaned["Volume"] = df_cleaned["Volume"].astype("float64")

        # Strip out any remaining non-numeric object columns if present
        object_cols = df_cleaned.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            logger.warning(f"Dropping unexpected object dtype columns: {object_cols}")
            df_cleaned = df_cleaned.drop(columns=object_cols)
            cols_dropped.extend(object_cols)

        mem_after = df_cleaned.memory_usage(deep=True).sum()
        logger.info(
            f"Data types optimized. Memory usage: "
            f"{mem_before / 1024:.2f} KB -> {mem_after / 1024:.2f} KB "
            f"({(1 - mem_after / mem_before) * 100:.2f}% savings)."
        )
        self.summary_report["memory_before_bytes"] = mem_before
        self.summary_report["memory_after_bytes"] = mem_after

        # Stage 7: Feature Engineering
        logger.info("Stage 7: Computing mathematical return and range features...")
        # 1. Daily Simple Return: R_t = P_t / P_{t-1} - 1
        df_cleaned["Simple_Return"] = df_cleaned["Close"].pct_change()
        
        # 2. Daily Log Return: ln(P_t / P_{t-1})
        df_cleaned["Log_Return"] = np.log(df_cleaned["Close"] / df_cleaned["Close"].shift(1))
        
        # 3. Daily Price Change: Close - Open
        df_cleaned["Price_Change"] = df_cleaned["Close"] - df_cleaned["Open"]
        
        # 4. Daily Range: High - Low
        df_cleaned["Daily_Range"] = df_cleaned["High"] - df_cleaned["Low"]
        
        # 5. Percentage Range: (High - Low) / Close
        df_cleaned["Pct_Range"] = (df_cleaned["High"] - df_cleaned["Low"]) / df_cleaned["Close"]
        
        features_created = ["Simple_Return", "Log_Return", "Price_Change", "Daily_Range", "Pct_Range"]
        logger.info(f"Engineered {len(features_created)} basic features: {features_created}")
        self.summary_report["features_created"] = features_created

        # Stage 8: Outlier Inspection (Do NOT delete outliers)
        logger.info(f"Stage 8: Inspecting returns for anomalies (threshold={self.outlier_threshold * 100:.1f}%)...")
        # Flag instances where absolute return exceeds the threshold
        outlier_mask = df_cleaned["Simple_Return"].abs() > self.outlier_threshold
        outliers_df = df_cleaned[outlier_mask].copy()
        num_outliers = len(outliers_df)
        
        logger.info(f"Detected {num_outliers} daily return outliers exceeding absolute threshold.")
        self.summary_report["outliers_count"] = num_outliers
        
        if num_outliers > 0:
            outlier_dates = outliers_df.index.strftime("%Y-%m-%d").tolist()
            logger.info(f"Outlier dates: {outlier_dates}")

        # Stage 9: Final Validation
        logger.info("Stage 9: Executing final clean dataset validations...")
        validate_clean_data(df_cleaned, keep_volume=self.keep_volume)
        logger.info("Final clean dataset validation passed successfully.")

        # Stage 10: Save Processed Dataset
        logger.info("Stage 10: Saving processed datasets to disk...")
        os.makedirs(os.path.dirname(clean_parquet_path), exist_ok=True)
        os.makedirs(os.path.dirname(clean_csv_path), exist_ok=True)
        
        try:
            df_cleaned.to_parquet(clean_parquet_path, index=True, engine="pyarrow")
            logger.info(f"Saved preprocessed parquet dataset to: {clean_parquet_path}")

            df_cleaned.to_csv(clean_csv_path, index=True)
            logger.info(f"Saved preprocessed CSV dataset to: {clean_csv_path}")
        except Exception as e:
            logger.error(f"Failed to serialize final clean datasets: {str(e)}")
            raise PreprocessingError(f"Error saving clean datasets: {str(e)}") from e

        # Stage 11: Data Quality Report
        self.summary_report["final_rows"] = len(df_cleaned)
        self.summary_report["final_cols"] = len(df_cleaned.columns)
        self.summary_report["final_date_range"] = (
            df_cleaned.index.min().strftime("%Y-%m-%d"),
            df_cleaned.index.max().strftime("%Y-%m-%d")
        )
        self.generate_and_log_quality_report()

        return df_cleaned, outliers_df

    def generate_and_log_quality_report(self) -> None:
        """
        Stage 11: Generates and logs a preprocessing data quality summary report.
        """
        initial_rows = self.summary_report.get("initial_rows", 0)
        final_rows = self.summary_report.get("final_rows", 0)
        removed_rows = self.summary_report.get("rows_removed_missing_or_duplicate", 0)
        outliers_count = self.summary_report.get("outliers_count", 0)
        
        mem_before = self.summary_report.get("memory_before_bytes", 0)
        mem_after = self.summary_report.get("memory_after_bytes", 0)
        
        initial_dates = self.summary_report.get("initial_date_range", ("N/A", "N/A"))
        final_dates = self.summary_report.get("final_date_range", ("N/A", "N/A"))

        logger.info("=============================================================")
        logger.info("               DATA PREPROCESSING SUMMARY                    ")
        logger.info("=============================================================")
        logger.info(f"Initial Dataset Shape : {initial_rows} rows x {self.summary_report.get('initial_cols', 0)} cols")
        logger.info(f"Final Dataset Shape   : {final_rows} rows x {self.summary_report.get('final_cols', 0)} cols")
        logger.info(f"Date Range (Initial)  : {initial_dates[0]} to {initial_dates[1]}")
        logger.info(f"Date Range (Final)    : {final_dates[0]} to {final_dates[1]}")
        logger.info(f"Rows Dropped (NaN/Dup): {removed_rows}")
        logger.info(f"Columns Dropped       : {self.summary_report.get('columns_dropped', [])}")
        logger.info(f"Features Created      : {self.summary_report.get('features_created', [])}")
        logger.info(f"Outliers Detected     : {outliers_count} observations")
        logger.info(f"Initial Memory Usage  : {mem_before / 1024:.2f} KB")
        logger.info(f"Final Memory Usage    : {mem_after / 1024:.2f} KB")
        logger.info(f"Memory Saved          : {(1 - mem_after / max(1, mem_before)) * 100:.2f}%")
        logger.info("=============================================================")
