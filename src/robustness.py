"""
Robustness and Sensitivity Analysis Module.

Implements parameter sensitivity grid construction, parameter neighborhood
evaluation, and computes stability and generalization metrics.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.validation import optimize_parameters

logger = logging.getLogger(__name__)

def generate_neighborhood_grid(
    strategy_type: str,
    best_params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generates a parameter grid of neighboring combinations around the best parameters.

    Args:
        strategy_type (str): 'momentum' or 'mean_reversion'.
        best_params (Dict[str, Any]): Dictionary of optimized parameters.

    Returns:
        List[Dict[str, Any]]: List of neighboring parameter combinations.
    """
    grid = []

    if strategy_type == "momentum":
        short_val = int(best_params["short_window"])
        long_val = int(best_params["long_window"])
        
        # Test s in [best-2, best-1, best, best+1, best+2]
        # Test l in [best-10, best-5, best, best+5, best+10]
        for s_offset in [-2, -1, 0, 1, 2]:
            for l_offset in [-10, -5, 0, 5, 10]:
                s = short_val + s_offset
                l = long_val + l_offset
                if s > 2 and l > s:
                    grid.append({"short_window": s, "long_window": l})
                    
    elif strategy_type == "mean_reversion":
        w_val = int(best_params["window"])
        entry_val = float(best_params["entry_threshold"])
        exit_val = float(best_params["exit_threshold"])
        
        # Test w in [best-2, best, best+2]
        # Test entry in [best-0.2, best, best+0.2]
        # Test exit in [best-0.1, best, best+0.1]
        for w_offset in [-2, 0, 2]:
            for entry_offset in [-0.2, 0.0, 0.2]:
                for exit_offset in [-0.1, 0.0, 0.1]:
                    w = w_val + w_offset
                    entry = entry_val + entry_offset
                    exit_p = exit_val + exit_offset
                    if w > 2 and entry < exit_p:
                        grid.append({
                            "window": w,
                            "entry_threshold": round(entry, 2),
                            "exit_threshold": round(exit_p, 2)
                        })
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")

    # Remove duplicates
    unique_grid = []
    seen = set()
    for params in grid:
        frozen = tuple(sorted(params.items()))
        if frozen not in seen:
            seen.add(frozen)
            unique_grid.append(params)

    logger.info(f"Generated {len(unique_grid)} neighborhood combinations for sensitivity analysis.")
    return unique_grid

def run_sensitivity_analysis(
    prices: pd.DataFrame,
    strategy_type: str,
    best_params: Dict[str, Any],
    initial_capital: float = 100000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    return_type: str = "simple",
    execution_type: str = "next_open"
) -> pd.DataFrame:
    """
    Evaluates neighboring parameters on the given prices (In-Sample period).

    Args:
        prices (pd.DataFrame): Price data.
        strategy_type (str): 'momentum' or 'mean_reversion'.
        best_params (Dict[str, Any]): Optimized parameters dict.
        initial_capital (float): Initial capital.
        transaction_cost_bps (float): Transaction costs.
        slippage_bps (float): Slippage.
        return_type (str): Returns style.
        execution_type (str): Execution model.

    Returns:
        pd.DataFrame: Sliced grid search results on the neighboring parameter sets.
    """
    grid = generate_neighborhood_grid(strategy_type, best_params)
    
    # Run optimization on this sub-grid
    # Sort results by parameter combination instead of rank to make plotting easy if needed,
    # but optimize_parameters returns ranked.
    sensitivity_results = optimize_parameters(
        prices=prices,
        strategy_type=strategy_type,
        param_grid=grid,
        objective="max_sharpe",
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        return_type=return_type,
        execution_type=execution_type
    )
    
    return sensitivity_results

def calculate_stability_score(
    sensitivity_df: pd.DataFrame,
    metric_col: str = "Sharpe_Ratio"
) -> float:
    """
    Computes a parameter stability score (0 to 100).

    A strategy is stable if its performance does not fluctuate widely across
    neighboring parameter sets.
    Formula:
        Score = 100 * (1 - min(1, Coefficient of Variation))
        Coefficient of Variation = Volatility of metric / Mean of metric
        If Mean <= 0, Stability is 0.0.

    Args:
        sensitivity_df (pd.DataFrame): Results of sensitivity analysis.
        metric_col (str): Column name of the metric to analyze. Default is 'Sharpe_Ratio'.

    Returns:
        float: Stability score (0 to 100).
    """
    if sensitivity_df.empty or metric_col not in sensitivity_df.columns:
        return 0.0

    values = sensitivity_df[metric_col]
    mean_val = values.mean()
    std_val = values.std(ddof=1)

    if pd.isna(std_val) or std_val == 0.0:
        return 100.0

    if mean_val <= 0.0:
        return 0.0

    cv = std_val / mean_val
    stability_score = (1.0 - min(1.0, cv)) * 100.0
    return float(stability_score)

def calculate_generalization_score(
    is_sharpe: float,
    oos_sharpe: float
) -> float:
    """
    Computes strategy generalization score based on Sharpe Ratio degradation.

    Formula:
        If OOS Sharpe >= IS Sharpe: 100.0 (Perfect Generalization)
        If OOS Sharpe <= 0: 0.0 (No Generalization / Strategy Fails)
        Otherwise: (OOS Sharpe / IS Sharpe) * 100.0

    Args:
        is_sharpe (float): Annualized Sharpe Ratio in-sample.
        oos_sharpe (float): Annualized Sharpe Ratio out-of-sample.

    Returns:
        float: Generalization score (0 to 100).
    """
    if is_sharpe <= 0.0:
        # If the optimized in-sample Sharpe was negative, there is no generalization baseline
        return 0.0 if oos_sharpe <= 0.0 else 100.0

    if oos_sharpe >= is_sharpe:
        return 100.0
    
    if oos_sharpe <= 0.0:
        return 0.0

    gen_score = (oos_sharpe / is_sharpe) * 100.0
    return float(gen_score)

def calculate_performance_degradation(
    is_val: float,
    oos_val: float,
    invert: bool = False
) -> float:
    """
    Calculates percentage performance degradation from IS to OOS.

    Formula:
        Degradation % = ((IS - OOS) / IS) * 100
        If invert is True (e.g. Drawdown where lower is better):
        Degradation % = ((OOS - IS) / IS) * 100

    Args:
        is_val (float): In-Sample metric value.
        oos_val (float): Out-of-Sample metric value.
        invert (bool): Invert values for metrics where lower is better. Default is False.

    Returns:
        float: Percentage change (can be positive for degradation, negative for improvement).
    """
    if is_val == 0.0:
        return 0.0

    if not invert:
        degradation = ((is_val - oos_val) / is_val) * 100.0
    else:
        degradation = ((oos_val - is_val) / is_val) * 100.0

    return float(degradation)
