"""
Execution Model Mapping Module.

Responsible for mapping raw strategy signals to portfolio position exposure states
based on execution models and timing rules (e.g. next-open or next-close execution).
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def map_signals_to_positions(signals: pd.Series, execution_type: str = "next_open") -> pd.Series:
    """
    Maps raw strategy signals to portfolio positions based on execution models and lags.

    Args:
        signals (pd.Series): Series of strategy signals (0.0 or 1.0, or NaN).
        execution_type (str): Execution timing model. Supported:
            - 'next_open': Signal generated today at close (t) is executed tomorrow at open (t+1).
                           Position for day t+1 is equal to raw signal at t.
                           Therefore, Position = Signals.shift(1).
            - 'next_close': Signal generated today at close (t) is executed tomorrow at close (t+1).
                            The first day we hold the position is day t+2.
                            Therefore, Position = Signals.shift(2).

    Returns:
        pd.Series: A series of position states (1.0 for Long, -1.0 for Short, 0.0 for Flat).
    """
    if not isinstance(signals, pd.Series):
        raise TypeError("Signals must be a pandas Series.")

    logger.info(f"Mapping signals to positions using execution model: '{execution_type}'")

    if execution_type == "next_open":
        position = signals.shift(1)
    elif execution_type == "next_close":
        position = signals.shift(2)
    else:
        raise ValueError(
            f"Unknown execution type: {execution_type}. "
            f"Supported options: 'next_open', 'next_close'"
        )

    # Initial shift period is filled with 0 (flat position)
    position = position.fillna(0.0)
    position.name = "Position"

    logger.debug(f"Mapped positions. Long days: {(position == 1.0).sum()}, Short days: {(position == -1.0).sum()}")

    return position
