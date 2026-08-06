"""
Transaction Costs and Slippage Modeling Module.

Provides pure mathematical functions to compute turnover, transaction costs,
and slippage in return space vectorially.
"""

import logging
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def calculate_turnover(position: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates entry and exit turnover vectorially based on position state changes.

    Turnover represents the absolute change in exposure. If a position changes sign
    (e.g., from +1 to -1), it is decomposed into an exit of size 1.0 and an entry of size 1.0.

    Args:
        position (pd.Series): Series of position states (1.0 for Long, -1.0 for Short, 0.0 for Flat).

    Returns:
        Tuple[pd.Series, pd.Series]:
            - entry_turnover (pd.Series): Entry turnover per day.
            - exit_turnover (pd.Series): Exit turnover per day.
    """
    if not isinstance(position, pd.Series):
        raise TypeError("Position must be a pandas Series.")

    pos_prev = position.shift(1).fillna(0.0)

    # Decompose into entry and exit turnovers
    same_sign = (position * pos_prev >= 0.0)

    entry_turnover = pd.Series(0.0, index=position.index, name="Entry_Turnover")
    exit_turnover = pd.Series(0.0, index=position.index, name="Exit_Turnover")

    # Same sign or transition to/from zero
    entry_turnover[same_sign] = np.maximum(0.0, position.abs() - pos_prev.abs())[same_sign]
    exit_turnover[same_sign] = np.maximum(0.0, pos_prev.abs() - position.abs())[same_sign]

    # Opposite signs (flips, e.g. -1 to 1 or 1 to -1)
    opp_sign = ~same_sign
    entry_turnover[opp_sign] = position[opp_sign].abs()
    exit_turnover[opp_sign] = pos_prev[opp_sign].abs()

    logger.debug(
        f"Calculated turnover. Total Entry Turnover: {entry_turnover.sum():.2f}, "
        f"Total Exit Turnover: {exit_turnover.sum():.2f}"
    )

    return entry_turnover, exit_turnover

def calculate_transaction_costs(
    position: pd.Series,
    cost_bps: float,
    slippage_bps: float,
    apply_on: str = "both"
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates the transaction cost return and slippage return impact daily.

    Args:
        position (pd.Series): Series of position states.
        cost_bps (float): Transaction cost in basis points (e.g. 5 bps = 0.0005).
        slippage_bps (float): Slippage in basis points (e.g. 2 bps = 0.0002).
        apply_on (str): When to apply transaction costs: 'entry', 'exit', or 'both'.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]:
            - transaction_cost_return (pd.Series): Cost returns (positive values to be subtracted).
            - slippage_return (pd.Series): Slippage returns (positive values to be subtracted).
            - total_cost_return (pd.Series): Combined cost return impact.
    """
    if cost_bps < 0 or slippage_bps < 0:
        raise ValueError("Cost and slippage basis points must be non-negative.")
    
    if apply_on not in ["entry", "exit", "both"]:
        raise ValueError(f"apply_on must be 'entry', 'exit', or 'both'. Got: {apply_on}")

    logger.info(
        f"Calculating transaction costs (cost_bps={cost_bps}, slippage_bps={slippage_bps}, apply_on='{apply_on}')"
    )

    entry_turnover, exit_turnover = calculate_turnover(position)

    cost_rate = cost_bps / 10000.0
    slippage_rate = slippage_bps / 10000.0

    if apply_on == "both":
        total_turnover = entry_turnover + exit_turnover
        tx_cost = total_turnover * cost_rate
        slippage = total_turnover * slippage_rate
    elif apply_on == "entry":
        tx_cost = entry_turnover * cost_rate
        slippage = entry_turnover * slippage_rate
    elif apply_on == "exit":
        tx_cost = exit_turnover * cost_rate
        slippage = exit_turnover * slippage_rate

    total_cost = tx_cost + slippage

    tx_cost.name = "Transaction_Cost_Return"
    slippage.name = "Slippage_Return"
    total_cost.name = "Total_Cost_Return"

    return tx_cost, slippage, total_cost
