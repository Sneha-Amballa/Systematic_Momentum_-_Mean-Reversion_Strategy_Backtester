import logging
import os
import sys
from src.utils import setup_logging, ensure_directories
from src.data_loader import download_ticker_data, validate_data, save_data
from src.preprocessing import DataPreprocessor
from src.eda import QuantEDA
from src.momentum import MomentumSignalGenerator

def main() -> None:
    """
    Main orchestration script for the Systematic Momentum & Mean-Reversion Strategy Backtester.
    Runs the pipeline steps:
      - Step 1: Data Acquisition (runs if raw data is missing)
      - Step 2: Data Cleaning & Preprocessing (runs always)
      - Step 3: Exploratory Data Analysis (runs always)
    """
    # 1. Initialize logging system
    setup_logging(logging.INFO)
    logger = logging.getLogger("main")
    logger.info("Initializing Backtesting Research Pipeline...")

    # Configuration Parameters
    ticker = "^NSEI"
    start_date = "2013-01-01"
    end_date = "2025-01-01"  # yfinance end parameter is exclusive; gets daily data up to 2024-12-31

    # Paths
    raw_csv_path = "data/raw/nifty50_raw.csv"
    raw_parquet_path = "data/raw/nifty50_raw.parquet"
    clean_csv_path = "data/processed/nifty50_clean.csv"
    clean_parquet_path = "data/processed/nifty50_clean.parquet"

    # Preprocessing Configurations
    keep_volume = False          # Volume is dropped by default as it is unreliable for indices
    outlier_threshold = 0.08     # Mark returns > 8% as outliers

    required_directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "reports/figures"
    ]

    try:
        # Ensure project directory structure exists
        ensure_directories(required_directories)

        # =====================================================================
        # STEP 1: Data Acquisition
        # =====================================================================
        if not os.path.exists(raw_parquet_path):
            logger.info("Raw dataset not found locally. Triggering Step 1: Data Acquisition...")
            raw_df = download_ticker_data(ticker=ticker, start_date=start_date, end_date=end_date)
            validate_data(raw_df)
            save_data(df=raw_df, csv_path=raw_csv_path, parquet_path=raw_parquet_path)
            logger.info("Step 1: Data Acquisition completed successfully.")
        else:
            logger.info(f"Raw dataset already exists at '{raw_parquet_path}'. Skipping acquisition step.")

        # =====================================================================
        # STEP 2: Data Cleaning & Preprocessing
        # =====================================================================
        logger.info("Triggering Step 2: Data Cleaning & Preprocessing...")
        preprocessor = DataPreprocessor(keep_volume=keep_volume, outlier_threshold=outlier_threshold)
        clean_df, outliers_df = preprocessor.process(
            raw_path=raw_parquet_path,
            clean_parquet_path=clean_parquet_path,
            clean_csv_path=clean_csv_path
        )
        logger.info("Step 2: Data Cleaning & Preprocessing completed successfully.")

        # =====================================================================
        # STEP 3: Exploratory Data Analysis (EDA)
        # =====================================================================
        logger.info("Triggering Step 3: Exploratory Data Analysis...")
        eda = QuantEDA(
            data_path=clean_parquet_path,
            figures_dir="reports/figures",
            report_path="reports/eda_summary.md"
        )
        eda.run_all()
        logger.info("Step 3: Exploratory Data Analysis completed successfully.")

        # =====================================================================
        # STEP 4A: Momentum Strategy Signal Construction
        # =====================================================================
        logger.info("Triggering Step 4A: Momentum Strategy Signal Construction...")
        
        # Paths for output
        momentum_parquet_path = "data/processed/momentum_signals.parquet"
        momentum_csv_path = "data/processed/momentum_signals.csv"
        figures_dir = "reports/figures"

        # Signal parameters
        short_window = 20
        long_window = 100

        # Initialize Momentum Signal Generator
        generator = MomentumSignalGenerator(short_window=short_window, long_window=long_window)

        # 1. Generate Signals
        logger.info("Generating indicators and strategy signals...")
        signals_df = generator.generate_signals(clean_df, close_col="Close")

        # 2. Detect Trades
        logger.info("Extracting trades from position states...")
        trades_df = generator.detect_trades(signals_df, position_col="Position")

        # 3. Compute Summary Statistics
        logger.info("Computing strategy summary metrics...")
        stats = generator.compute_statistics(signals_df, trades_df)

        # Log statistics professionally
        logger.info("=== MOMENTUM STRATEGY SIGNAL STATISTICS ===")
        logger.info(f"Total Active Trading Days           : {stats.get('Total_Active_Trading_Days')}")
        logger.info(f"Total Raw Buy Days (Short > Long)   : {stats.get('Total_Raw_Buy_Days')}")
        logger.info(f"Total Raw Sell Days (Short <= Long) : {stats.get('Total_Raw_Sell_Days')}")
        logger.info(f"Total Execution Buy Days            : {stats.get('Total_Execution_Buy_Days')}")
        logger.info(f"Total Execution Sell Days           : {stats.get('Total_Execution_Sell_Days')}")
        logger.info(f"Number of Trade Entries (Transitions) : {stats.get('Number_of_Trades')}")
        logger.info(f"Average Holding Period (Trading Days) : {stats.get('Average_Holding_Period_Trading_Days'):.2f}")
        logger.info(f"Maximum Holding Period (Trading Days) : {stats.get('Maximum_Holding_Period_Trading_Days')}")
        logger.info(f"Minimum Holding Period (Trading Days) : {stats.get('Minimum_Holding_Period_Trading_Days')}")
        logger.info(f"Average Holding Period (Calendar Days): {stats.get('Average_Holding_Period_Calendar_Days'):.2f}")
        logger.info(f"Maximum Holding Period (Calendar Days): {stats.get('Maximum_Holding_Period_Calendar_Days')}")
        logger.info(f"Minimum Holding Period (Calendar Days): {stats.get('Minimum_Holding_Period_Calendar_Days')}")
        logger.info(f"Percentage of Time Invested         : {stats.get('Percentage_of_Time_Invested'):.2f}%")
        logger.info(f"Flat Market Percentage              : {stats.get('Flat_Market_Percentage'):.2f}%")
        logger.info("===========================================")

        # 4. Save Signal Output
        logger.info("Saving momentum signal datasets...")
        signals_df.to_parquet(momentum_parquet_path, index=True, engine="pyarrow")
        signals_df.to_csv(momentum_csv_path, index=True)
        logger.info(f"Saved signals Parquet to: {momentum_parquet_path}")
        logger.info(f"Saved signals CSV to: {momentum_csv_path}")

        # Save Trades table for completeness
        trades_csv_path = "data/processed/momentum_trades.csv"
        trades_df.to_csv(trades_csv_path, index=False)
        logger.info(f"Saved detected trades log CSV to: {trades_csv_path}")

        # 5. Generate Professional Visualizations
        logger.info("Generating publication-quality charts...")
        generator.plot_performance(signals_df, trades_df, output_dir=figures_dir)
        
        logger.info("Step 4A: Momentum Strategy Signal Construction completed successfully.")
        logger.info("Pipeline executed successfully!")

    except Exception as e:
        logger.critical(f"Pipeline execution halted due to error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
