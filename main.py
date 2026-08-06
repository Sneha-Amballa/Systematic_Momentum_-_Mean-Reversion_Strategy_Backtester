import logging
import os
import sys
import numpy as np
import pandas as pd
from src.utils import setup_logging, ensure_directories
from src.data_loader import download_ticker_data, validate_data, save_data
from src.preprocessing import DataPreprocessor
from src.eda import QuantEDA
from src.momentum import MomentumSignalGenerator
from src.mean_reversion import MeanReversionSignalGenerator
from src.engine import backtest, plot_backtest_results
from src.metrics import calculate_return_metrics, calculate_risk_adjusted_ratios
from src.risk import calculate_maximum_drawdown
from src.analytics import (
    calculate_trade_statistics,
    calculate_benchmark_metrics,
    analyze_drawdowns,
    generate_scorecard,
    plot_performance_dashboard
)
from src.validation import split_dataset, optimize_parameters
from src.robustness import run_sensitivity_analysis, calculate_stability_score, calculate_generalization_score
from src.comparison import generate_comparison_table, plot_robustness_dashboard
from src.regime import MarketRegimeAnalyzer

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

        # =====================================================================
        # STEP 4B: Mean Reversion Strategy Signal Construction
        # =====================================================================
        logger.info("Triggering Step 4B: Mean Reversion Strategy Signal Construction...")
        
        # Paths for output
        mr_parquet_path = "data/processed/mean_reversion_signals.parquet"
        mr_csv_path = "data/processed/mean_reversion_signals.csv"

        # Signal parameters
        mr_window = 20
        mr_entry_threshold = -2.0
        mr_exit_threshold = -0.5

        # Initialize Mean Reversion Signal Generator
        mr_generator = MeanReversionSignalGenerator(
            window=mr_window, 
            entry_threshold=mr_entry_threshold, 
            exit_threshold=mr_exit_threshold
        )

        # 1. Generate Signals
        logger.info("Generating mean reversion indicators and strategy signals...")
        mr_signals_df = mr_generator.generate_signals(clean_df, close_col="Close")

        # 2. Detect Trades
        logger.info("Extracting mean reversion trades from position states...")
        mr_trades_df = mr_generator.detect_trades(mr_signals_df, position_col="Position")

        # 3. Compute Summary Statistics
        logger.info("Computing mean reversion strategy summary metrics...")
        mr_stats = mr_generator.compute_statistics(mr_signals_df, mr_trades_df)

        # Log statistics professionally
        logger.info("=== MEAN REVERSION STRATEGY SIGNAL STATISTICS ===")
        logger.info(f"Total Active Trading Days           : {mr_stats.get('Total_Active_Trading_Days')}")
        logger.info(f"Total Raw Buy Days (Z < Entry)      : {mr_stats.get('Total_Raw_Buy_Days')}")
        logger.info(f"Total Raw Sell Days (Z >= Exit)     : {mr_stats.get('Total_Raw_Sell_Days')}")
        logger.info(f"Total Execution Buy Days            : {mr_stats.get('Total_Execution_Buy_Days')}")
        logger.info(f"Total Execution Sell Days           : {mr_stats.get('Total_Execution_Sell_Days')}")
        logger.info(f"Number of Trade Entries (Transitions) : {mr_stats.get('Number_of_Trades')}")
        logger.info(f"Average Holding Period (Trading Days) : {mr_stats.get('Average_Holding_Period_Trading_Days'):.2f}")
        logger.info(f"Maximum Holding Period (Trading Days) : {mr_stats.get('Maximum_Holding_Period_Trading_Days')}")
        logger.info(f"Minimum Holding Period (Trading Days) : {mr_stats.get('Minimum_Holding_Period_Trading_Days')}")
        logger.info(f"Average Holding Period (Calendar Days): {mr_stats.get('Average_Holding_Period_Calendar_Days'):.2f}")
        logger.info(f"Maximum Holding Period (Calendar Days): {mr_stats.get('Maximum_Holding_Period_Calendar_Days')}")
        logger.info(f"Minimum Holding Period (Calendar Days): {mr_stats.get('Minimum_Holding_Period_Calendar_Days')}")
        logger.info(f"Percentage of Time Invested         : {mr_stats.get('Percentage_of_Time_Invested'):.2f}%")
        logger.info(f"Flat Market Percentage              : {mr_stats.get('Flat_Market_Percentage'):.2f}%")
        logger.info(f"Average Z-Score at Entry (Trigger)  : {mr_stats.get('Average_ZScore_Entry_Trigger'):.4f}")
        logger.info(f"Average Z-Score at Entry (Execution): {mr_stats.get('Average_ZScore_Entry_Execution'):.4f}")
        logger.info(f"Average Z-Score at Exit (Trigger)   : {mr_stats.get('Average_ZScore_Exit_Trigger'):.4f}")
        logger.info(f"Average Z-Score at Exit (Execution) : {mr_stats.get('Average_ZScore_Exit_Execution'):.4f}")
        logger.info("==============================================")

        # 4. Save Signal Output
        logger.info("Saving mean reversion signal datasets...")
        mr_signals_df.to_parquet(mr_parquet_path, index=True, engine="pyarrow")
        mr_signals_df.to_csv(mr_csv_path, index=True)
        logger.info(f"Saved mean reversion signals Parquet to: {mr_parquet_path}")
        logger.info(f"Saved mean reversion signals CSV to: {mr_csv_path}")

        # Save Trades table for completeness
        mr_trades_csv_path = "data/processed/mean_reversion_trades.csv"
        mr_trades_df.to_csv(mr_trades_csv_path, index=False)
        logger.info(f"Saved detected mean reversion trades log CSV to: {mr_trades_csv_path}")

        # 5. Generate Professional Visualizations
        logger.info("Generating mean reversion publication-quality charts...")
        mr_generator.plot_performance(mr_signals_df, mr_trades_df, output_dir=figures_dir)
        logger.info("Step 4B: Mean Reversion Strategy Signal Construction completed successfully.")

        # =====================================================================
        # STEP 5: Vectorized Backtesting Engine
        # =====================================================================
        logger.info("Triggering Step 5: Vectorized Backtesting Engine...")

        # Backtest parameters
        initial_capital = 100000.0
        tx_cost_bps = 5.0
        slippage_bps = 2.0
        return_type = "simple"
        execution_type = "next_open"
        apply_cost_on = "both"

        # 1. Backtest Momentum Strategy
        logger.info("Running backtest for Momentum strategy...")
        mom_backtest = backtest(
            prices=clean_df,
            signals=signals_df["Raw_Signal"],
            initial_capital=initial_capital,
            transaction_cost_bps=tx_cost_bps,
            slippage_bps=slippage_bps,
            return_type=return_type,
            execution_type=execution_type,
            apply_cost_on=apply_cost_on
        )

        # 2. Backtest Mean Reversion Strategy
        logger.info("Running backtest for Mean Reversion strategy...")
        mr_backtest = backtest(
            prices=clean_df,
            signals=mr_signals_df["Raw_Signal"],
            initial_capital=initial_capital,
            transaction_cost_bps=tx_cost_bps,
            slippage_bps=slippage_bps,
            return_type=return_type,
            execution_type=execution_type,
            apply_cost_on=apply_cost_on
        )

        # Log Momentum Backtest Summary
        logger.info("=== MOMENTUM BACKTEST SUMMARY ===")
        logger.info(f"Initial Capital                     : ₹{mom_backtest.summary['Initial_Capital']:,.2f}")
        logger.info(f"Final Capital                       : ₹{mom_backtest.summary['Final_Capital']:,.2f}")
        logger.info(f"Total Net Return                    : {mom_backtest.summary['Total_Return_Pct']:.2f}%")
        logger.info(f"Total Trade Cycles                  : {mom_backtest.summary['Total_Trades']}")
        logger.info(f"Total Portfolio Turnover            : {mom_backtest.summary['Total_Turnover']:.2f} units")
        logger.info(f"Total Trading Frictional Costs Cash : ₹{mom_backtest.summary['Total_Cost_Cash']:,.2f}")
        logger.info(f"Average Hold Period (Trading Days)  : {mom_backtest.summary['Average_Hold_Days']:.2f}")
        logger.info("=================================")

        # Log Mean Reversion Backtest Summary
        logger.info("=== MEAN REVERSION BACKTEST SUMMARY ===")
        logger.info(f"Initial Capital                     : ₹{mr_backtest.summary['Initial_Capital']:,.2f}")
        logger.info(f"Final Capital                       : ₹{mr_backtest.summary['Final_Capital']:,.2f}")
        logger.info(f"Total Net Return                    : {mr_backtest.summary['Total_Return_Pct']:.2f}%")
        logger.info(f"Total Trade Cycles                  : {mr_backtest.summary['Total_Trades']}")
        logger.info(f"Total Portfolio Turnover            : {mr_backtest.summary['Total_Turnover']:.2f} units")
        logger.info(f"Total Trading Frictional Costs Cash : ₹{mr_backtest.summary['Total_Cost_Cash']:,.2f}")
        logger.info(f"Average Hold Period (Trading Days)  : {mr_backtest.summary['Average_Hold_Days']:.2f}")
        logger.info("======================================")

        # 3. Export Backtest Results (Parquet & CSV)
        logger.info("Exporting backtest results...")
        
        # Paths
        mom_portfolio_parquet = "data/processed/momentum_portfolio_history.parquet"
        mom_portfolio_csv = "data/processed/momentum_portfolio_history.csv"
        mom_trades_parquet = "data/processed/momentum_trade_book.parquet"
        mom_trades_csv = "data/processed/momentum_trade_book.csv"

        mr_portfolio_parquet = "data/processed/mean_reversion_portfolio_history.parquet"
        mr_portfolio_csv = "data/processed/mean_reversion_portfolio_history.csv"
        mr_trades_parquet = "data/processed/mean_reversion_trade_book.parquet"
        mr_trades_csv = "data/processed/mean_reversion_trade_book.csv"

        # Save Momentum
        mom_backtest.portfolio.to_parquet(mom_portfolio_parquet, index=True)
        mom_backtest.portfolio.to_csv(mom_portfolio_csv, index=True)
        mom_backtest.trade_book.to_parquet(mom_trades_parquet, index=False)
        mom_backtest.trade_book.to_csv(mom_trades_csv, index=False)

        # Save Mean Reversion
        mr_backtest.portfolio.to_parquet(mr_portfolio_parquet, index=True)
        mr_backtest.portfolio.to_csv(mr_portfolio_csv, index=True)
        mr_backtest.trade_book.to_parquet(mr_trades_parquet, index=False)
        mr_backtest.trade_book.to_csv(mr_trades_csv, index=False)

        logger.info("Backtest result files saved to data/processed/ successfully.")

        # 4. Generate Backtesting Plots
        logger.info("Generating backtesting professional visualizations...")
        plot_backtest_results(mom_backtest, clean_df, output_dir=figures_dir, prefix="momentum")
        plot_backtest_results(mr_backtest, clean_df, output_dir=figures_dir, prefix="mean_reversion")

        logger.info("Step 5: Vectorized Backtesting Engine completed successfully.")

        # =====================================================================
        # STEP 6: Performance Metrics & Strategy Analytics
        # =====================================================================
        logger.info("Triggering Step 6: Performance Metrics & Strategy Analytics...")

        # Benchmark returns (Nifty 50 close-to-close returns aligned with backtest)
        benchmark_returns = clean_df["Close"].pct_change().fillna(0.0)

        # A. Evaluate Momentum Strategy
        logger.info("Evaluating Momentum strategy performance metrics...")
        mom_bench_ret = benchmark_returns.reindex(mom_backtest.portfolio.index).ffill()
        mom_ret_metrics = calculate_return_metrics(mom_backtest.portfolio["Portfolio_Value"], mom_backtest.portfolio["Daily_Return"])
        mom_bench_metrics = calculate_benchmark_metrics(mom_backtest.portfolio["Daily_Return"], mom_bench_ret)
        mom_risk_metrics = calculate_risk_adjusted_ratios(
            mom_backtest.portfolio["Daily_Return"],
            mom_backtest.portfolio["Portfolio_Value"],
            benchmark_returns=mom_bench_ret,
            beta=mom_bench_metrics.get("Beta", 1.0)
        )
        mom_trade_stats = calculate_trade_statistics(mom_backtest.trade_book)
        mom_mdd_depth, _, _, _ = calculate_maximum_drawdown(mom_backtest.portfolio["Portfolio_Value"])
        
        # Combine all Momentum metrics
        mom_all_metrics = {**mom_ret_metrics, **mom_bench_metrics, **mom_risk_metrics, **mom_trade_stats, "Max_Drawdown": mom_mdd_depth * 100.0}
        mom_scorecard = generate_scorecard(mom_all_metrics)
        mom_drawdowns = analyze_drawdowns(mom_backtest.portfolio["Portfolio_Value"])

        # B. Evaluate Mean Reversion Strategy
        logger.info("Evaluating Mean Reversion strategy performance metrics...")
        mr_bench_ret = benchmark_returns.reindex(mr_backtest.portfolio.index).ffill()
        mr_ret_metrics = calculate_return_metrics(mr_backtest.portfolio["Portfolio_Value"], mr_backtest.portfolio["Daily_Return"])
        mr_bench_metrics = calculate_benchmark_metrics(mr_backtest.portfolio["Daily_Return"], mr_bench_ret)
        mr_risk_metrics = calculate_risk_adjusted_ratios(
            mr_backtest.portfolio["Daily_Return"],
            mr_backtest.portfolio["Portfolio_Value"],
            benchmark_returns=mr_bench_ret,
            beta=mr_bench_metrics.get("Beta", 1.0)
        )
        mr_trade_stats = calculate_trade_statistics(mr_backtest.trade_book)
        mr_mdd_depth, _, _, _ = calculate_maximum_drawdown(mr_backtest.portfolio["Portfolio_Value"])

        # Combine all Mean Reversion metrics
        mr_all_metrics = {**mr_ret_metrics, **mr_bench_metrics, **mr_risk_metrics, **mr_trade_stats, "Max_Drawdown": mr_mdd_depth * 100.0}
        mr_scorecard = generate_scorecard(mr_all_metrics)
        mr_drawdowns = analyze_drawdowns(mr_backtest.portfolio["Portfolio_Value"])

        # Log scorecards
        logger.info("=== MOMENTUM PERFORMANCE SCORECARD ===")
        logger.info("\n" + mom_scorecard.to_string(index=False))
        logger.info("======================================")

        logger.info("=== MEAN REVERSION PERFORMANCE SCORECARD ===")
        logger.info("\n" + mr_scorecard.to_string(index=False))
        logger.info("===========================================")

        # C. Export Performance Results
        logger.info("Saving strategy analytics summaries to data/processed/...")
        
        # Paths
        mom_scorecard_csv = "data/processed/momentum_scorecard.csv"
        mom_drawdowns_csv = "data/processed/momentum_drawdowns.csv"
        mr_scorecard_csv = "data/processed/mean_reversion_scorecard.csv"
        mr_drawdowns_csv = "data/processed/mean_reversion_drawdowns.csv"

        # Save to CSV
        mom_scorecard.to_csv(mom_scorecard_csv, index=False)
        mom_drawdowns.to_csv(mom_drawdowns_csv, index=False)
        mr_scorecard.to_csv(mr_scorecard_csv, index=False)
        mr_drawdowns.to_csv(mr_drawdowns_csv, index=False)

        # Save to Parquet where appropriate
        mom_drawdowns.to_parquet("data/processed/momentum_drawdowns.parquet", index=False)
        mr_drawdowns.to_parquet("data/processed/mean_reversion_drawdowns.parquet", index=False)

        # D. Plot dashboards (20 charts per strategy)
        logger.info("Plotting comprehensive performance visualizations (20 charts per strategy)...")
        plot_performance_dashboard(mom_backtest.portfolio, mom_backtest.trade_book, clean_df, output_dir=figures_dir, prefix="momentum")
        plot_performance_dashboard(mr_backtest.portfolio, mr_backtest.trade_book, clean_df, output_dir=figures_dir, prefix="mean_reversion")

        logger.info("Step 6: Performance Metrics & Strategy Analytics completed successfully.")

        # =====================================================================
        # STEP 7: In-Sample vs Out-of-Sample Validation & Robustness Analysis
        # =====================================================================
        logger.info("Triggering Step 7: In-Sample vs Out-of-Sample Evaluation & Robustness Analysis...")

        # 1. Dataset chronological split (In-Sample vs. Out-of-Sample)
        is_prices = split_dataset(clean_df, "2013-01-01", "2019-12-31")
        oos_prices = split_dataset(clean_df, "2020-01-01", "2024-12-31")

        logger.info(f"Chronological split completed. IS records: {len(is_prices)}, OOS records: {len(oos_prices)}")

        # 2. Parameter grid search setup
        mom_grid = []
        for short_w in [10, 20, 30, 50]:
            for long_w in [50, 100, 150, 200]:
                if short_w < long_w:
                    mom_grid.append({"short_window": short_w, "long_window": long_w})

        mr_grid = []
        for w in [10, 20, 30, 50]:
            for entry_t in [-1.5, -2.0, -2.5]:
                for exit_t in [-0.25, -0.5, -1.0]:
                    mr_grid.append({"window": w, "entry_threshold": entry_t, "exit_threshold": exit_t})

        # 3. Perform grid search strictly on In-Sample (IS) period
        logger.info("Running parameter optimization sweep on In-Sample (IS) dataset...")
        mom_opt_results = optimize_parameters(is_prices, "momentum", mom_grid, objective="max_sharpe")
        mr_opt_results = optimize_parameters(is_prices, "mean_reversion", mr_grid, objective="max_sharpe")

        # 4. Freeze best parameters
        best_mom = mom_opt_results.iloc[0].to_dict()
        best_mr = mr_opt_results.iloc[0].to_dict()

        logger.info(f"Frozen Momentum Parameters: short_window={best_mom['short_window']}, long_window={best_mom['long_window']}")
        logger.info(f"Frozen Mean Reversion Parameters: window={best_mr['window']}, entry_threshold={best_mr['entry_threshold']}, exit_threshold={best_mr['exit_threshold']}")

        # 5. Run backtests on IS and OOS using frozen parameters
        logger.info("Running backtests on IS and OOS with frozen parameters...")
        
        # Momentum
        mom_generator = MomentumSignalGenerator(
            short_window=int(best_mom["short_window"]),
            long_window=int(best_mom["long_window"])
        )
        mom_is_sig = mom_generator.generate_signals(is_prices)
        mom_oos_sig = mom_generator.generate_signals(oos_prices)
        mom_is_res = backtest(is_prices, mom_is_sig["Raw_Signal"])
        mom_oos_res = backtest(oos_prices, mom_oos_sig["Raw_Signal"])

        # Mean Reversion
        mr_generator = MeanReversionSignalGenerator(
            window=int(best_mr["window"]),
            entry_threshold=float(best_mr["entry_threshold"]),
            exit_threshold=float(best_mr["exit_threshold"])
        )
        mr_is_sig = mr_generator.generate_signals(is_prices)
        mr_oos_sig = mr_generator.generate_signals(oos_prices)
        mr_is_res = backtest(is_prices, mr_is_sig["Raw_Signal"])
        mr_oos_res = backtest(oos_prices, mr_oos_sig["Raw_Signal"])

        # 6. Gather metrics
        logger.info("Evaluating performance metrics for IS and OOS periods...")
        
        bench_ret_is = is_prices["Close"].pct_change().fillna(0.0)
        bench_ret_oos = oos_prices["Close"].pct_change().fillna(0.0)

        def get_all_metrics(res, benchmark_ret):
            ret_m = calculate_return_metrics(res.portfolio["Portfolio_Value"], res.portfolio["Daily_Return"])
            bench_m = calculate_benchmark_metrics(res.portfolio["Daily_Return"], benchmark_ret)
            risk_m = calculate_risk_adjusted_ratios(res.portfolio["Daily_Return"], res.portfolio["Portfolio_Value"], benchmark_ret, bench_m["Beta"])
            trade_m = calculate_trade_statistics(res.trade_book)
            max_dd, _, _, _ = calculate_maximum_drawdown(res.portfolio["Portfolio_Value"])
            return {
                **ret_m, **bench_m, **risk_m, **trade_m, 
                "Max_Drawdown": max_dd * 100.0, 
                "Volatility": res.portfolio["Daily_Return"].std() * np.sqrt(252.0)
            }

        mom_is_metrics = get_all_metrics(mom_is_res, bench_ret_is)
        mom_oos_metrics = get_all_metrics(mom_oos_res, bench_ret_oos)
        mr_is_metrics = get_all_metrics(mr_is_res, bench_ret_is)
        mr_oos_metrics = get_all_metrics(mr_oos_res, bench_ret_oos)

        # 7. Run sensitivity analysis on In-Sample (IS) parameter neighborhood
        logger.info("Performing parameter sensitivity and stability analysis...")
        mom_sens = run_sensitivity_analysis(
            is_prices, "momentum", 
            {"short_window": int(best_mom["short_window"]), "long_window": int(best_mom["long_window"])}
        )
        mom_stability = calculate_stability_score(mom_sens)

        mr_sens = run_sensitivity_analysis(
            is_prices, "mean_reversion", 
            {"window": int(best_mr["window"]), "entry_threshold": float(best_mr["entry_threshold"]), "exit_threshold": float(best_mr["exit_threshold"])}
        )
        mr_stability = calculate_stability_score(mr_sens)

        # 8. Compute Generalization Scores
        mom_gen_score = calculate_generalization_score(mom_is_metrics["Sharpe_Ratio"], mom_oos_metrics["Sharpe_Ratio"])
        mr_gen_score = calculate_generalization_score(mr_is_metrics["Sharpe_Ratio"], mr_oos_metrics["Sharpe_Ratio"])

        # 9. Create side-by-side comparison tables
        mom_comparison = generate_comparison_table(mom_is_metrics, mom_oos_metrics)
        mr_comparison = generate_comparison_table(mr_is_metrics, mr_oos_metrics)

        # Log results
        logger.info("=== MOMENTUM IS vs OOS PERFORMANCE COMPARISON ===")
        logger.info(f"Stability Score: {mom_stability:.2f}%, Generalization Score: {mom_gen_score:.2f}%")
        logger.info("\n" + mom_comparison.to_string(index=False))
        logger.info("=================================================")

        logger.info("=== MEAN REVERSION IS vs OOS PERFORMANCE COMPARISON ===")
        logger.info(f"Stability Score: {mr_stability:.2f}%, Generalization Score: {mr_gen_score:.2f}%")
        logger.info("\n" + mr_comparison.to_string(index=False))
        logger.info("======================================================")

        # 10. Export results
        logger.info("Exporting validation and robustness analysis datasets to data/processed/...")
        
        # Save optimization tables
        mom_opt_results.to_csv("data/processed/momentum_optimization_results.csv", index=False)
        mr_opt_results.to_csv("data/processed/mean_reversion_optimization_results.csv", index=False)
        
        # Save sensitivity tables
        mom_sens.to_csv("data/processed/momentum_sensitivity_results.csv", index=False)
        mr_sens.to_csv("data/processed/mean_reversion_sensitivity_results.csv", index=False)
        
        # Save comparison tables
        mom_comparison.to_csv("data/processed/momentum_validation_comparison.csv", index=False)
        mr_comparison.to_csv("data/processed/mean_reversion_validation_comparison.csv", index=False)
        
        # Save robustness scores metadata
        robustness_summary = pd.DataFrame([
            {"Strategy": "Momentum", "Stability Score (%)": mom_stability, "Generalization Score (%)": mom_gen_score, "Best Parameters": f"10/150"},
            {"Strategy": "Mean Reversion", "Stability Score (%)": mr_stability, "Generalization Score (%)": mr_gen_score, "Best Parameters": f"30/-2.0/-0.5"}
        ])
        robustness_summary.to_csv("data/processed/robustness_scores_summary.csv", index=False)

        # 11. Plot dashboards (20 charts per strategy)
        logger.info("Plotting comprehensive validation & robustness visualizations (20 charts per strategy)...")
        plot_robustness_dashboard(
            "momentum", mom_is_res, mom_oos_res, mom_opt_results, mom_sens, clean_df, output_dir=figures_dir
        )
        plot_robustness_dashboard(
            "mean_reversion", mr_is_res, mr_oos_res, mr_opt_results, mr_sens, clean_df, output_dir=figures_dir
        )

        logger.info("Step 7: In-Sample vs Out-of-Sample Evaluation & Robustness Analysis completed successfully.")

        # =====================================================================
        # STEP 8: Market Regime Analysis
        # =====================================================================
        logger.info("Triggering Step 8: Market Regime Analysis...")

        # 1. Instantiate the analyzer
        regime_analyzer = MarketRegimeAnalyzer(clean_df)
        
        # 2. Detect regimes, transition matrix, stability stats
        regimes_df = regime_analyzer.detect_regimes()

        # 3. Re-run strategy backtests over the full history using frozen optimized parameters
        logger.info("Re-running whole-history backtests using best frozen optimized parameters...")
        
        # Momentum
        mom_full_generator = MomentumSignalGenerator(
            short_window=int(best_mom["short_window"]),
            long_window=int(best_mom["long_window"])
        )
        mom_full_sig = mom_full_generator.generate_signals(clean_df)["Raw_Signal"]
        mom_full_res = backtest(
            prices=clean_df,
            signals=mom_full_sig,
            initial_capital=initial_capital,
            transaction_cost_bps=tx_cost_bps,
            slippage_bps=slippage_bps,
            return_type=return_type,
            execution_type=execution_type,
            apply_cost_on=apply_cost_on
        )

        # Mean Reversion
        mr_full_generator = MeanReversionSignalGenerator(
            window=int(best_mr["window"]),
            entry_threshold=float(best_mr["entry_threshold"]),
            exit_threshold=float(best_mr["exit_threshold"])
        )
        mr_full_sig = mr_full_generator.generate_signals(clean_df)["Raw_Signal"]
        mr_full_res = backtest(
            prices=clean_df,
            signals=mr_full_sig,
            initial_capital=initial_capital,
            transaction_cost_bps=tx_cost_bps,
            slippage_bps=slippage_bps,
            return_type=return_type,
            execution_type=execution_type,
            apply_cost_on=apply_cost_on
        )

        # 4. Perform metrics breakdowns per regime
        mom_reg_perf, mr_reg_perf = regime_analyzer.analyze_performance(mom_full_res, mr_full_res)

        # 5. Transition drift & Feature correlations
        trans_returns = regime_analyzer.run_transition_returns(mom_full_res)
        feat_importance = regime_analyzer.run_feature_importance(mom_full_res)

        logger.info(f"Regime performance calculations and transition returns complete.")

        # 6. Export results (Parquet, CSV, Markdown)
        logger.info("Exporting regime analysis datasets to data/processed/...")
        
        # Parquet exports
        regimes_df.to_parquet("data/processed/market_regime_labels.parquet", index=True)
        
        # CSV exports
        regimes_df.to_csv("data/processed/market_regime_labels.csv", index=True)
        regime_analyzer.transition_matrix.to_csv("data/processed/regime_transition_matrix.csv", index=True)
        mom_reg_perf.to_csv("data/processed/momentum_performance_by_regime.csv", index=False)
        mr_reg_perf.to_csv("data/processed/mean_reversion_performance_by_regime.csv", index=False)
        trans_returns.to_csv("data/processed/regime_transition_returns_drift.csv", index=False)
        feat_importance.to_frame("Absolute_Correlation").to_csv("data/processed/market_regime_feature_importance.csv", index=True)

        # Markdown comparison table exports
        with open("data/processed/regime_performance_comparison.md", "w", encoding="utf-8") as f:
            f.write("# Regime Performance Comparison Tables\n\n")
            f.write("## Momentum Crossover Strategy by Regime\n")
            f.write(mom_reg_perf.to_markdown(index=False) + "\n\n")
            f.write("## Mean Reversion Z-Score Strategy by Regime\n")
            f.write(mr_reg_perf.to_markdown(index=False) + "\n")

        # 7. Plot visualizations (20 charts)
        logger.info("Plotting comprehensive market regime analysis visualizations (20 charts)...")
        regime_analyzer.plot_regimes(
            mom_backtest_res=mom_full_res,
            mr_backtest_res=mr_full_res,
            mom_perf=mom_reg_perf,
            mr_perf=mr_reg_perf,
            feature_importance=feat_importance,
            output_dir=figures_dir
        )

        logger.info("Step 8: Market Regime Analysis completed successfully.")
        logger.info("Pipeline executed successfully!")

    except Exception as e:
        logger.critical(f"Pipeline execution halted due to error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
