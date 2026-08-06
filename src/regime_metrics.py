"""
Regime Performance Metrics Module.

Calculates strategy return/risk metrics partitioned by regime, compiles
Markov transition matrices, computes post-transition return schedules,
and ranks market feature correlations.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.risk import calculate_downside_volatility, calculate_maximum_drawdown, calculate_drawdown_series
from src.metrics import calculate_return_metrics, calculate_risk_adjusted_ratios

logger = logging.getLogger(__name__)

def calculate_regime_performance(
    portfolio_df: pd.DataFrame,
    trade_book: pd.DataFrame,
    regimes: pd.Series,
    annualization_factor: float = 252.0
) -> pd.DataFrame:
    """
    Computes performance metrics partitioned by distinct market regimes.

    Calculations are done on synthetic contiguous return series compiled for each regime.

    Args:
        portfolio_df (pd.DataFrame): Daily portfolio value and returns history.
        trade_book (pd.DataFrame): Trade book log.
        regimes (pd.Series): Daily regime labels series.
        annualization_factor (float): Daily periods in a year.

    Returns:
        pd.DataFrame: Performance metrics by regime table.
    """
    unique_regimes = regimes.dropna().unique()
    rows = []

    for regime_name in unique_regimes:
        # Filter daily data
        regime_mask = regimes == regime_name
        if not regime_mask.any():
            continue

        returns_regime = portfolio_df.loc[regime_mask, "Daily_Return"]
        n_days = len(returns_regime)
        if n_days <= 2:
            continue

        # 1. Compile synthetic contiguous equity curve for this regime
        synthetic_equity = (1.0 + returns_regime).cumprod() * 100000.0
        
        # 2. Return metrics
        ret_metrics = calculate_return_metrics(synthetic_equity, returns_regime, annualization_factor)
        
        # 3. Risk metrics
        vol = returns_regime.std(ddof=1) * np.sqrt(annualization_factor)
        max_dd, _, _, _ = calculate_maximum_drawdown(synthetic_equity)
        
        # 4. Ratios
        risk_ratios = calculate_risk_adjusted_ratios(
            returns=returns_regime,
            portfolio_value=synthetic_equity,
            annualization_factor=annualization_factor
        )

        # 5. Position Exposure & Turnover
        # Exposure is percentage of days where absolute position > 0
        positions_regime = portfolio_df.loc[regime_mask, "Position"] if "Position" in portfolio_df.columns else pd.Series(0, index=returns_regime.index)
        exposure_pct = (positions_regime.abs() > 0).sum() / n_days * 100.0
        
        # Turnover is sum of absolute diffs of position
        # Sliced daily differences inside the regime
        turnover_regime = positions_regime.diff().fillna(0.0).abs().sum()

        # 6. Trade statistics in regime
        # Find trades that entered during this regime
        regime_dates = returns_regime.index
        if not trade_book.empty and "Entry Date" in trade_book.columns:
            trades_in_regime = trade_book[trade_book["Entry Date"].isin(regime_dates)]
            trade_count = len(trades_in_regime)
            
            if trade_count > 0:
                gross_returns = trades_in_regime["Position Size"] * (trades_in_regime["Exit Price"] - trades_in_regime["Entry Price"]) / trades_in_regime["Entry Price"]
                cost_returns = trades_in_regime["Total Cost"] / (100000.0 * trades_in_regime["Position Size"].abs())
                net_trade_returns = gross_returns - cost_returns
                
                win_rate = (net_trade_returns > 0).sum() / trade_count * 100.0
                
                winners = net_trade_returns[net_trade_returns > 0]
                losers = net_trade_returns[net_trade_returns <= 0]
                profit_factor = winners.sum() / abs(losers.sum()) if losers.sum() != 0.0 else np.nan
                
                avg_hold = float(trades_in_regime["Holding Days"].mean())
                avg_trade_ret = float(net_trade_returns.mean()) * 100.0
            else:
                win_rate = np.nan
                profit_factor = np.nan
                avg_hold = 0.0
                avg_trade_ret = 0.0
        else:
            trade_count = 0
            win_rate = np.nan
            profit_factor = np.nan
            avg_hold = 0.0
            avg_trade_ret = 0.0

        rows.append({
            "Regime": regime_name,
            "Active Days": n_days,
            "CAGR (%)": ret_metrics.get("CAGR", 0.0) * 100.0,
            "Volatility (%)": vol * 100.0,
            "Sharpe Ratio": risk_ratios.get("Sharpe_Ratio", 0.0),
            "Sortino Ratio": risk_ratios.get("Sortino_Ratio", 0.0),
            "Calmar Ratio": risk_ratios.get("Calmar_Ratio", 0.0),
            "Max Drawdown (%)": max_dd * 100.0,
            "Trade Count": trade_count,
            "Win Rate (%)": win_rate,
            "Profit Factor": profit_factor,
            "Average Hold (TD)": avg_hold,
            "Avg Trade Return (%)": avg_trade_ret,
            "Exposure (%)": exposure_pct,
            "Turnover (Units)": turnover_regime
        })

    return pd.DataFrame(rows).sort_values(by="Regime").reset_index(drop=True)

def calculate_regime_stability(regimes: pd.Series) -> Dict[str, Any]:
    """
    Computes statistical regime persistence and durations.

    RLE approach is used to isolate contiguous blocks.

    Args:
        regimes (pd.Series): Daily regime labels series.

    Returns:
        Dict[str, Any]: Stability statistics dictionary.
    """
    if regimes.empty:
        return {}

    # Run Length Encoding: Identify contiguous regime blocks
    blocks = (regimes != regimes.shift()).cumsum()
    block_lengths = regimes.groupby(blocks).size()
    block_names = regimes.groupby(blocks).first()

    durations_df = pd.DataFrame({
        "Regime": block_names,
        "Duration": block_lengths
    })

    stability_stats = {
        "Total_Regime_Changes": int(len(block_lengths) - 1),
        "Average_Duration": float(block_lengths.mean()),
        "Max_Duration": int(block_lengths.max()),
        "Min_Duration": int(block_lengths.min())
    }

    # Summary by regime type
    by_regime = durations_df.groupby("Regime")["Duration"].agg(["mean", "max", "min", "count"])
    by_regime.columns = ["Avg Duration", "Max Duration", "Min Duration", "Occurrences"]
    
    stability_stats["By_Regime"] = by_regime.to_dict(orient="index")
    return stability_stats

def calculate_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    Computes Markov transition probability matrix between regime states.

    Matrix elements P_ij represent probability of transitioning from state i to j on day t+1.

    Args:
        regimes (pd.Series): Daily regime labels.

    Returns:
        pd.DataFrame: Markov transition probability matrix.
    """
    df = pd.DataFrame({
        "Current": regimes,
        "Next": regimes.shift(-1)
    }).dropna()

    if df.empty:
        return pd.DataFrame()

    # Crosstab absolute counts
    counts = pd.crosstab(df["Current"], df["Next"])
    
    # Standardize row sums to 1.0 (conditional probabilities)
    transition_matrix = counts.div(counts.sum(axis=1), axis=0)
    
    return transition_matrix

def calculate_transition_returns(
    portfolio_df: pd.DataFrame,
    regimes: pd.Series,
    windows: List[int] = [30, 60, 90]
) -> pd.DataFrame:
    """
    Tracks portfolio performance drift following regime transitions.

    Args:
        portfolio_df (pd.DataFrame): Daily strategy history.
        regimes (pd.Series): Daily regime labels.
        windows (List[int]): Days forward to check. Default [30, 60, 90].

    Returns:
        pd.DataFrame: Average returns post transition.
    """
    # Find transition indexes (regime[t] != regime[t-1])
    diffs = regimes != regimes.shift(1)
    transition_idx = diffs[diffs].index
    
    # Exclude the very first index
    if len(transition_idx) > 0 and transition_idx[0] == regimes.index[0]:
        transition_idx = transition_idx[1:]

    records = []

    for t_date in transition_idx:
        current_loc = regimes.index.get_loc(t_date)
        prev_regime = regimes.iloc[current_loc - 1]
        new_regime = regimes.iloc[current_loc]
        
        transition_name = f"{prev_regime} -> {new_regime}"
        
        # Calculate returns forward
        rets = {}
        for w in windows:
            end_loc = min(current_loc + w, len(portfolio_df) - 1)
            fwd_returns = portfolio_df["Daily_Return"].iloc[current_loc:end_loc]
            # Cumulative return
            cum_ret = (1.0 + fwd_returns).prod() - 1.0
            rets[f"Return_{w}d (%)"] = float(cum_ret) * 100.0

        records.append({
            "Transition Date": t_date,
            "Transition": transition_name,
            **rets
        })

    df = pd.DataFrame(records)
    if df.empty:
        cols = ["Transition"] + [f"Return_{w}d (%)" for w in windows]
        return pd.DataFrame(columns=cols)

    # Average by transition type
    avg_df = df.groupby("Transition")[[f"Return_{w}d (%)" for w in windows]].mean().reset_index()
    return avg_df

def calculate_feature_importance(
    prices: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    window: int = 20,
    close_col: str = "Close"
) -> pd.Series:
    """
    Computes absolute correlation between market characteristics and strategy returns.

    No ML models are used. It ranks features by Pearson correlation coefficient.

    Features evaluated:
        1. Volatility: Realized rolling volatility of the index.
        2. Trend Strength: (50-day SMA - 200-day SMA) / 200-day SMA.
        3. Index Drawdown: Drawdown from peak.
        4. Market Direction: Daily returns of Nifty 50.
        5. Rolling Return: 20-day return of Nifty 50.

    Args:
        prices (pd.DataFrame): Index pricing.
        portfolio_df (pd.DataFrame): Daily strategy history.
        window (int): Realized vol rolling window.
        close_col (str): Price column.

    Returns:
        pd.Series: Ranks and absolute correlations.
    """
    close = prices[close_col]
    
    # 1. Volatility
    daily_idx_ret = close.pct_change()
    realized_vol = daily_idx_ret.rolling(window).std() * np.sqrt(252.0)
    
    # 2. Trend Strength
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    trend_strength = (sma_50 - sma_200) / sma_200

    # 3. Drawdown
    drawdown = calculate_drawdown_series(close)

    # 4. Market Direction
    mkt_dir = daily_idx_ret

    # 5. Rolling Return
    rolling_ret = close.pct_change(window)

    features_df = pd.DataFrame({
        "Index Volatility": realized_vol,
        "Trend Strength": trend_strength,
        "Index Drawdown": drawdown,
        "Index Daily Return": mkt_dir,
        "Index Rolling Return": rolling_ret
    }, index=prices.index).bfill().ffill()

    # Align with strategy returns
    aligned = pd.concat([features_df, portfolio_df["Daily_Return"]], axis=1).dropna()
    feat_sliced = aligned.iloc[:, :-1]
    strat_ret = aligned.iloc[:, -1]

    # Calculate correlation
    correlations = feat_sliced.corrwith(strat_ret).abs().sort_values(ascending=False)
    
    return correlations
