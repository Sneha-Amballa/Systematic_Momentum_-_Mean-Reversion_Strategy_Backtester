"""
Market Regime Analysis Orchestrator.

Provides the MarketRegimeAnalyzer class to coordinate
regime classification, metrics extraction, stability logs, and visualizations.
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd

from src.regime_detector import detect_combined_regimes
from src.regime_metrics import (
    calculate_regime_performance,
    calculate_regime_stability,
    calculate_transition_matrix,
    calculate_transition_returns,
    calculate_feature_importance
)
from src.regime_visualization import plot_regime_analysis

logger = logging.getLogger(__name__)

class MarketRegimeAnalyzer:
    """
    Orchestrates the entire market regime analysis workflow for multiple trading strategies.
    """

    def __init__(self, prices: pd.DataFrame, close_col: str = "Close"):
        """
        Initializes the MarketRegimeAnalyzer.

        Args:
            prices (pd.DataFrame): Daily index pricing data.
            close_col (str): Column name for close prices. Default is 'Close'.
        """
        self.prices = prices
        self.close_col = close_col
        self.regimes_df = None
        self.transition_matrix = None
        self.stability_stats = None

    def detect_regimes(self) -> pd.DataFrame:
        """
        Classifies historical trend, volatility, and combined crash/recovery regimes.

        Returns:
            pd.DataFrame: Regime labels table.
        """
        logger.info("Triggering objective market regime classification...")
        self.regimes_df = detect_combined_regimes(self.prices, self.close_col)
        
        # Stability stats
        self.stability_stats = calculate_regime_stability(self.regimes_df["Final_Regime"])
        
        # Transition matrix
        self.transition_matrix = calculate_transition_matrix(self.regimes_df["Final_Regime"])
        
        return self.regimes_df

    def analyze_performance(
        self,
        mom_backtest_res: Any,
        mr_backtest_res: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes performance statistics for both strategies separately inside each regime.

        Args:
            mom_backtest_res (BacktestResult): Momentum backtest output.
            mr_backtest_res (BacktestResult): Mean Reversion backtest output.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - mom_reg_perf (pd.DataFrame): Momentum stats by regime.
                - mr_reg_perf (pd.DataFrame): Mean Reversion stats by regime.
        """
        if self.regimes_df is None:
            self.detect_regimes()

        logger.info("Computing strategy performance breakdowns per market regime...")
        
        # We align regimes to the specific portfolio dates
        mom_regimes = self.regimes_df["Final_Regime"].reindex(mom_backtest_res.portfolio.index).ffill()
        mr_regimes = self.regimes_df["Final_Regime"].reindex(mr_backtest_res.portfolio.index).ffill()

        mom_reg_perf = calculate_regime_performance(
            mom_backtest_res.portfolio,
            mom_backtest_res.trade_book,
            mom_regimes
        )
        
        mr_reg_perf = calculate_regime_performance(
            mr_backtest_res.portfolio,
            mr_backtest_res.trade_book,
            mr_regimes
        )

        return mom_reg_perf, mr_reg_perf

    def run_transition_returns(self, mom_backtest_res: Any) -> pd.DataFrame:
        """
        Evaluates portfolio returns drift following regime transition events.

        Args:
            mom_backtest_res (BacktestResult): Backtest results.

        Returns:
            pd.DataFrame: Returns drift table.
        """
        if self.regimes_df is None:
            self.detect_regimes()
        return calculate_transition_returns(mom_backtest_res.portfolio, self.regimes_df["Final_Regime"])

    def run_feature_importance(self, mom_backtest_res: Any) -> pd.Series:
        """
        Computes absolute correlation between market features and strategy returns.

        Args:
            mom_backtest_res (BacktestResult): Strategy results.

        Returns:
            pd.Series: Correlation values.
        """
        return calculate_feature_importance(self.prices, mom_backtest_res.portfolio)

    def plot_regimes(
        self,
        mom_backtest_res: Any,
        mr_backtest_res: Any,
        mom_perf: pd.DataFrame,
        mr_perf: pd.DataFrame,
        feature_importance: pd.Series,
        output_dir: str
    ) -> None:
        """
        Generates and saves the 20 regime-colored, transition, and dashboard charts.

        Args:
            mom_backtest_res (BacktestResult): Momentum backtest result.
            mr_backtest_res (BacktestResult): Mean Reversion backtest result.
            mom_perf (pd.DataFrame): Momentum performance by regime.
            mr_perf (pd.DataFrame): Mean Reversion performance by regime.
            feature_importance (pd.Series): Feature correlation rankings.
            output_dir (str): output directory.
        """
        if self.regimes_df is None:
            self.detect_regimes()

        plot_regime_analysis(
            self.prices,
            self.regimes_df,
            mom_perf,
            mr_perf,
            self.transition_matrix,
            self.stability_stats,
            feature_importance,
            output_dir
        )
