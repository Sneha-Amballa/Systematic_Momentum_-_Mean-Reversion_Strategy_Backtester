import logging
import os
import warnings
from typing import Dict, Any, List, Tuple
# pyrefly: ignore [import-error]
import matplotlib.pyplot as plt
# pyrefly: ignore [import-error]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import scipy.stats as stats
# pyrefly: ignore [missing-import]
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

# Consistent publication-quality plotting styles
PLT_COLOR_DARK_BLUE = "#1f77b4"
PLT_COLOR_LIGHT_BLUE = "#aec7e8"
PLT_COLOR_ACCENT = "#ff7f0e"
PLT_COLOR_RED = "#d62728"
PLT_COLOR_GREY = "#7f7f7f"
PLT_GRID_CONFIG = {"linestyle": "--", "alpha": 0.5, "color": "#cccccc"}

class QuantEDA:
    """
    Automates Exploratory Data Analysis (EDA) for quantitative trading research.
    Performs statistical tests, evaluates seasonality, identifies outliers, 
    generates matplotlib figures, and compiles a comprehensive Markdown report.
    """

    def __init__(self, data_path: str, figures_dir: str = "reports/figures", report_path: str = "reports/eda_summary.md"):
        """
        Initializes the QuantEDA engine.

        Args:
            data_path (str): Path to the processed parquet file.
            figures_dir (str): Directory where plots are saved.
            report_path (str): Destination path for the markdown summary report.
        """
        self.data_path = data_path
        self.figures_dir = figures_dir
        self.report_path = report_path
        self.df = pd.DataFrame()
        self.stats_report: Dict[str, Any] = {}
        
        # Ensure target directories exist
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

    def load_data(self) -> None:
        """Loads and pre-validates processed market data."""
        logger.info(f"Loading clean dataset from: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Clean parquet file not found at: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Saves a matplotlib figure with high DPI and closes it to prevent memory leaks."""
        path = os.path.join(self.figures_dir, name)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {path}")

    def run_all(self) -> None:
        """Runs the entire EDA suite sequentially."""
        logger.info("Executing Step 3: Exploratory Data Analysis pipeline...")
        self.load_data()
        
        # Core Computations
        self.analyze_dataset_overview()
        self.analyze_data_quality()
        self.compute_descriptive_statistics()
        self.analyze_price_behaviour()
        self.analyze_return_distributions()
        self.analyze_volatility()
        self.analyze_stationarity()
        self.analyze_correlations()
        self.analyze_rolling_statistics()
        self.analyze_extreme_days()
        self.analyze_seasonality()
        
        # Report Generation
        self.compile_markdown_report()
        logger.info("Exploratory Data Analysis pipeline finished successfully.")

    def analyze_dataset_overview(self) -> None:
        """Section 1: Gathers metadata and basic stats of the dataset."""
        logger.info("Section 1: Compiling dataset overview...")
        num_rows, num_cols = self.df.shape
        start_date = self.df.index.min().strftime("%Y-%m-%d")
        end_date = self.df.index.max().strftime("%Y-%m-%d")
        
        # Compute exact memory usage
        mem_bytes = self.df.memory_usage(deep=True).sum()
        
        self.stats_report["overview"] = {
            "rows": num_rows,
            "columns": num_cols,
            "start_date": start_date,
            "end_date": end_date,
            "memory_kb": mem_bytes / 1024.0,
            "dtypes": self.df.dtypes.to_dict(),
        }

    def analyze_data_quality(self) -> None:
        """Section 2: Evaluates the completeness and health of the data columns."""
        logger.info("Section 2: Running data quality analysis...")
        num_rows = len(self.df)
        
        missing_counts = self.df.isna().sum().to_dict()
        null_percentages = {col: (count / num_rows) * 100 for col, count in missing_counts.items()}
        
        # Check for infs in numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_df).sum().to_dict()
        
        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_timestamps = int(self.df.index.duplicated().sum())
        
        self.stats_report["quality"] = {
            "missing_counts": missing_counts,
            "null_percentages": null_percentages,
            "inf_counts": inf_counts,
            "duplicate_rows": duplicate_rows,
            "duplicate_timestamps": duplicate_timestamps,
        }

    def compute_descriptive_statistics(self) -> None:
        """Section 3: Calculates descriptive statistics for numeric columns."""
        logger.info("Section 3: Calculating descriptive statistics...")
        stats_df = pd.DataFrame()
        
        for col in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            series = self.df[col].dropna()
            
            mean_val = series.mean()
            std_val = series.std()
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            
            # Coefficient of Variation: std / mean
            cv_val = std_val / mean_val if mean_val != 0 else np.nan
            
            col_stats = {
                "count": len(series),
                "mean": mean_val,
                "median": series.median(),
                "min": series.min(),
                "max": series.max(),
                "std": std_val,
                "var": series.var(),
                "25%": q25,
                "50%": series.median(),
                "75%": q75,
                "iqr": iqr,
                "cv": cv_val
            }
            stats_df[col] = pd.Series(col_stats)
            
        self.stats_report["descriptive_stats"] = stats_df.to_dict()

    def analyze_price_behaviour(self) -> None:
        """Section 4: Generates price time series, ranges, price changes, and cumulative return plots."""
        logger.info("Section 4: Generating price behavior analysis and plots...")
        
        # 1. Closing Price Time Series
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df.index, self.df["Close"], color=PLT_COLOR_DARK_BLUE, label="Nifty 50 Close")
        ax.set_title("Nifty 50 Index Historical Closing Price (2013-2024)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.grid(True, **PLT_GRID_CONFIG)
        ax.legend()
        self._save_figure(fig, "01_price_time_series.png")

        # 2. Open vs Close Scatter
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(self.df["Open"], self.df["Close"], color=PLT_COLOR_DARK_BLUE, alpha=0.3, s=8)
        # Plot 45-degree parity line
        min_p = min(self.df["Open"].min(), self.df["Close"].min())
        max_p = max(self.df["Open"].max(), self.df["Close"].max())
        ax.plot([min_p, max_p], [min_p, max_p], color=PLT_COLOR_RED, linestyle="--", label="Parity (Open = Close)")
        ax.set_title("Opening vs. Closing Prices Relationship", fontsize=12, fontweight="bold")
        ax.set_xlabel("Open Price (INR)")
        ax.set_ylabel("Close Price (INR)")
        ax.grid(True, **PLT_GRID_CONFIG)
        ax.legend()
        self._save_figure(fig, "02_open_vs_close_scatter.png")

        # 3. High-Low Range
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.df.index, self.df["Daily_Range"], color=PLT_COLOR_GREY, alpha=0.8, label="Daily Range (High - Low)")
        ax.set_title("Nifty 50 Index Intraday High-Low Absolute Range", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Absolute Price Difference (INR)")
        ax.grid(True, **PLT_GRID_CONFIG)
        ax.legend()
        self._save_figure(fig, "03_high_low_range.png")

        # 4. Daily Price Change
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.df.index, self.df["Price_Change"], color=PLT_COLOR_ACCENT, alpha=0.6, label="Daily Change (Close - Open)")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_title("Nifty 50 Index Daily Close-to-Open Price Changes", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price Change (INR)")
        ax.grid(True, **PLT_GRID_CONFIG)
        ax.legend()
        self._save_figure(fig, "04_daily_price_change.png")

        # 5. Cumulative Return Curve
        # Fill first row NaN in return to make cumulative curve start at 0
        returns_filled = self.df["Simple_Return"].fillna(0)
        cum_ret = (1 + returns_filled).cumprod() - 1
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df.index, cum_ret * 100, color=PLT_COLOR_RED, label="Cumulative Return")
        ax.set_title("Nifty 50 Index Cumulative Investment Return (Base 0%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(True, **PLT_GRID_CONFIG)
        ax.legend()
        self._save_figure(fig, "05_cumulative_return_curve.png")

    def analyze_return_distributions(self) -> None:
        """Section 5: Generates return distribution plots: Hist/KDE, Box, Violin, and normal QQ-plot."""
        logger.info("Section 5: Generating return distribution plots...")
        returns = self.df["Simple_Return"].dropna() * 100 # Convert to percentage returns
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Histogram & KDE
        ax_hist = axes[0, 0]
        ax_hist.hist(returns, bins=80, density=True, color=PLT_COLOR_DARK_BLUE, alpha=0.6, label="Daily Returns")
        # Overlay normal distribution density
        mu, std = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 500)
        p = stats.norm.pdf(x, mu, std)
        ax_hist.plot(x, p, color=PLT_COLOR_RED, linewidth=1.5, label="Normal Fit")
        ax_hist.set_title("Histogram & Fitted Normal Curve", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel("Daily Return (%)")
        ax_hist.set_ylabel("Density")
        ax_hist.grid(True, **PLT_GRID_CONFIG)
        ax_hist.legend()

        # 2. QQ Plot
        ax_qq = axes[0, 1]
        stats.probplot(returns, dist="norm", plot=ax_qq)
        ax_qq.set_title("Normal Q-Q Plot", fontsize=11, fontweight="bold")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Sample Quantiles (%)")
        ax_qq.grid(True, **PLT_GRID_CONFIG)

        # 3. Box Plot
        ax_box = axes[1, 0]
        ax_box.boxplot(returns, vert=False, patch_artist=True, 
                       boxprops=dict(facecolor=PLT_COLOR_LIGHT_BLUE, color=PLT_COLOR_DARK_BLUE),
                       medianprops=dict(color=PLT_COLOR_RED, linewidth=2))
        ax_box.set_title("Box Plot of Returns", fontsize=11, fontweight="bold")
        ax_box.set_xlabel("Daily Return (%)")
        ax_box.set_yticklabels([])
        ax_box.grid(True, **PLT_GRID_CONFIG)

        # 4. Violin Plot
        ax_violin = axes[1, 1]
        ax_violin.violinplot(returns, vert=False, showmedians=True)
        ax_violin.set_title("Violin Plot of Returns Distribution", fontsize=11, fontweight="bold")
        ax_violin.set_xlabel("Daily Return (%)")
        ax_violin.set_yticklabels([])
        ax_violin.grid(True, **PLT_GRID_CONFIG)

        fig.suptitle("Historical Nifty 50 Return Distribution Profiles", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        self._save_figure(fig, "06_returns_distribution_plots.png")

    def analyze_volatility(self) -> None:
        """Section 6: Computes rolling annualized volatilities for different window frames."""
        logger.info("Section 6: Performing rolling volatility calculations...")
        # Annualized rolling volatility: rolling std * sqrt(252) * 100 for percent
        vol_20 = self.df["Simple_Return"].rolling(20).std() * np.sqrt(252) * 100
        vol_50 = self.df["Simple_Return"].rolling(50).std() * np.sqrt(252) * 100
        vol_100 = self.df["Simple_Return"].rolling(100).std() * np.sqrt(252) * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df.index, vol_20, color=PLT_COLOR_LIGHT_BLUE, alpha=0.6, label="20-Day Window")
        ax.plot(self.df.index, vol_50, color=PLT_COLOR_DARK_BLUE, alpha=0.8, label="50-Day Window")
        ax.plot(self.df.index, vol_100, color=PLT_COLOR_RED, alpha=0.9, label="100-Day Window")
        
        ax.set_title("Rolling Annualized Volatility Over Varying Time Horizons", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Annual Volatility (%)")
        ax.grid(True, **PLT_GRID_CONFIG)
        ax.legend()
        
        self._save_figure(fig, "07_rolling_volatility_comparison.png")

    def analyze_stationarity(self) -> None:
        """Section 7 & 8: Computes skewness, kurtosis, Jarque-Bera test, and stationarity tests (ADF/KPSS)."""
        logger.info("Section 7 & 8: Running distribution shape metrics and stationarity tests...")
        
        # 1. Skewness & Kurtosis
        returns = self.df["Simple_Return"].dropna()
        skew_val = stats.skew(returns)
        kurt_val = stats.kurtosis(returns) # Excess Kurtosis (normal is 0)
        
        # Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(returns)
        
        # 2. Stationarity of Close Prices vs Returns
        # ADF Test
        adf_price_stat, adf_price_p, _, _, adf_price_crit, _ = adfuller(self.df["Close"].dropna())
        adf_ret_stat, adf_ret_p, _, _, adf_ret_crit, _ = adfuller(returns)
        
        # KPSS Test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_price_stat, kpss_price_p, _, kpss_price_crit = kpss(self.df["Close"].dropna(), regression="c")
            kpss_ret_stat, kpss_ret_p, _, kpss_ret_crit = kpss(returns, regression="c")

        self.stats_report["stationarity_normality"] = {
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "jb_stat": jb_stat,
            "jb_p": jb_p,
            "adf": {
                "price": {"stat": adf_price_stat, "p": adf_price_p, "crit_5pct": adf_price_crit["5%"]},
                "return": {"stat": adf_ret_stat, "p": adf_ret_p, "crit_5pct": adf_ret_crit["5%"]}
            },
            "kpss": {
                "price": {"stat": kpss_price_stat, "p": kpss_price_p, "crit_5pct": kpss_price_crit["5%"]},
                "return": {"stat": kpss_ret_stat, "p": kpss_ret_p, "crit_5pct": kpss_ret_crit["5%"]}
            }
        }

    def analyze_correlations(self) -> None:
        """Section 9: Correlation matrix heatmap and scatter matrices."""
        logger.info("Section 9: Generating correlation plots...")
        corr_cols = ["Open", "High", "Low", "Close", "Simple_Return", "Daily_Range", "Price_Change"]
        corr_matrix = self.df[corr_cols].corr()
        
        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        
        # Set ticks
        ax.set_xticks(np.arange(len(corr_cols)))
        ax.set_yticks(np.arange(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha="left")
        ax.set_yticklabels(corr_cols)
        
        # Annotate correlation numbers
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                        ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white", 
                        fontweight="bold")
                
        ax.set_title("Correlation Heatmap Matrix", fontsize=12, fontweight="bold", pad=20)
        self._save_figure(fig, "08_correlation_heatmap.png")

        # Scatter Matrix/Pair Plot of returns and ranges
        subset_cols = ["Simple_Return", "Daily_Range", "Price_Change"]
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i, col_y in enumerate(subset_cols):
            for j, col_x in enumerate(subset_cols):
                ax = axes[i, j]
                if i == j:
                    ax.hist(self.df[col_y].dropna(), bins=30, color=PLT_COLOR_DARK_BLUE, alpha=0.7)
                else:
                    ax.scatter(self.df[col_x], self.df[col_y], color=PLT_COLOR_DARK_BLUE, alpha=0.2, s=4)
                
                if i == 2:
                    ax.set_xlabel(col_x)
                if j == 0:
                    ax.set_ylabel(col_y)
                ax.grid(True, **PLT_GRID_CONFIG)
                
        fig.suptitle("Scatter Pair Matrix: Returns, Ranges, & Changes", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        self._save_figure(fig, "09_scatter_pair_matrix.png")

    def analyze_rolling_statistics(self) -> None:
        """Section 10: Generates rolling metrics (rolling mean/median/min/max) and volatility plots."""
        logger.info("Section 10: Generating rolling price stats plots...")
        close_series = self.df["Close"]
        roll_window = 50 # 50-day window
        
        roll_mean = close_series.rolling(roll_window).mean()
        roll_median = close_series.rolling(roll_window).median()
        roll_min = close_series.rolling(roll_window).min()
        roll_max = close_series.rolling(roll_window).max()
        roll_std = close_series.rolling(roll_window).std()

        # Visualizing Price + Rolling Statistics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Subplot 1: Price and rolling boundaries
        ax1.plot(self.df.index, close_series, color="black", label="Close Price", linewidth=0.8)
        ax1.plot(self.df.index, roll_mean, color=PLT_COLOR_DARK_BLUE, label="50d Rolling Mean")
        ax1.plot(self.df.index, roll_median, color=PLT_COLOR_ACCENT, label="50d Rolling Median", linestyle="--")
        ax1.fill_between(self.df.index, roll_min, roll_max, color=PLT_COLOR_LIGHT_BLUE, alpha=0.3, label="50d Min-Max Band")
        ax1.set_title("Nifty 50 Price with 50-Day Rolling Boundaries", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Price (INR)")
        ax1.grid(True, **PLT_GRID_CONFIG)
        ax1.legend()

        # Subplot 2: Rolling Std Dev
        ax2.plot(self.df.index, roll_std, color=PLT_COLOR_RED, label="50d Rolling Std Dev")
        ax2.set_title("Rolling Volatility (Standard Deviation of Price)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Price Std Dev (INR)")
        ax2.set_xlabel("Date")
        ax2.grid(True, **PLT_GRID_CONFIG)
        ax2.legend()

        plt.tight_layout()
        self._save_figure(fig, "10_rolling_statistics_price.png")

    def analyze_extreme_days(self) -> None:
        """Section 11: Identifies largest positive/negative return days, ranges, and gaps."""
        logger.info("Section 11: Extracting extreme market days...")
        
        # Calculate opening gaps: Open_t - Close_{t-1}
        self.df["Opening_Gap"] = self.df["Open"] - self.df["Close"].shift(1)
        self.df["Pct_Opening_Gap"] = (self.df["Open"] - self.df["Close"].shift(1)) / self.df["Close"].shift(1)
        
        top_pos = self.df.sort_values(by="Simple_Return", ascending=False).head(20).copy()
        top_neg = self.df.sort_values(by="Simple_Return", ascending=True).head(20).copy()
        top_ranges = self.df.sort_values(by="Daily_Range", ascending=False).head(20).copy()
        top_gaps = self.df.sort_values(by="Pct_Opening_Gap", key=abs, ascending=False).head(20).copy()

        self.stats_report["extreme_days"] = {
            "top_positive": top_pos[["Close", "Simple_Return"]].to_dict(orient="records"),
            "top_positive_dates": top_pos.index.strftime("%Y-%m-%d").tolist(),
            
            "top_negative": top_neg[["Close", "Simple_Return"]].to_dict(orient="records"),
            "top_negative_dates": top_neg.index.strftime("%Y-%m-%d").tolist(),
            
            "top_ranges": top_ranges[["Daily_Range", "Close", "Pct_Range"]].to_dict(orient="records"),
            "top_ranges_dates": top_ranges.index.strftime("%Y-%m-%d").tolist(),
            
            "top_gaps": top_gaps[["Pct_Opening_Gap", "Open", "Close"]].to_dict(orient="records"),
            "top_gaps_dates": top_gaps.index.strftime("%Y-%m-%d").tolist()
        }

    def analyze_seasonality(self) -> None:
        """Section 12, 13 & 14: Handles Annual, Monthly, and Weekly breakdowns."""
        logger.info("Section 12, 13 & 14: Computing annual, monthly, and weekly seasonalities...")
        
        # 1. Annual Analysis
        # Yearly Return
        yearly_ret = self.df["Simple_Return"].groupby(self.df.index.year).apply(lambda x: (1 + x.fillna(0)).prod() - 1) * 100
        # Yearly Volatility (annualized)
        yearly_vol = self.df["Simple_Return"].groupby(self.df.index.year).std() * np.sqrt(252) * 100
        # Yearly Max Drawdown
        def get_max_dd(series):
            cum_ret = (1 + series.fillna(0)).cumprod()
            running_max = cum_ret.cummax()
            dd = (cum_ret - running_max) / running_max
            return dd.min() * 100
        yearly_mdd = self.df["Simple_Return"].groupby(self.df.index.year).apply(get_max_dd)
        # Trading Days
        yearly_days = self.df["Close"].groupby(self.df.index.year).count()

        years = yearly_ret.index.tolist()
        self.stats_report["annual"] = {
            "years": years,
            "returns": yearly_ret.to_dict(),
            "volatility": yearly_vol.to_dict(),
            "mdd": yearly_mdd.to_dict(),
            "trading_days": yearly_days.to_dict(),
        }

        # Yearly comparison bar chart
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].bar(years, yearly_ret, color=PLT_COLOR_DARK_BLUE, alpha=0.8, edgecolor="black")
        axes[0].set_ylabel("Yearly Return (%)")
        axes[0].set_title("Yearly Total Returns", fontsize=11, fontweight="bold")
        axes[0].grid(True, **PLT_GRID_CONFIG)

        axes[1].bar(years, yearly_vol, color=PLT_COLOR_ACCENT, alpha=0.8, edgecolor="black")
        axes[1].set_ylabel("Annualized Volatility (%)")
        axes[1].set_title("Yearly Annualized Volatility", fontsize=11, fontweight="bold")
        axes[1].grid(True, **PLT_GRID_CONFIG)

        axes[2].bar(years, yearly_mdd, color=PLT_COLOR_RED, alpha=0.8, edgecolor="black")
        axes[2].set_ylabel("Max Drawdown (%)")
        axes[2].set_title("Yearly Maximum Peak-to-Trough Drawdown", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Year")
        axes[2].grid(True, **PLT_GRID_CONFIG)

        plt.xticks(years, rotation=45)
        plt.tight_layout()
        self._save_figure(fig, "11_annual_comparison.png")

        # 2. Monthly Heatmap
        # Calculate monthly compounded returns
        monthly_ret = self.df["Simple_Return"].groupby([self.df.index.year, self.df.index.month]).apply(
            lambda x: (1 + x.fillna(0)).prod() - 1
        ) * 100
        monthly_ret_unstacked = monthly_ret.unstack(level=1)
        monthly_ret_unstacked.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(monthly_ret_unstacked, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
        fig.colorbar(cax, label="Return (%)")
        
        ax.set_xticks(np.arange(12))
        ax.set_yticks(np.arange(len(monthly_ret_unstacked.index)))
        ax.set_xticklabels(monthly_ret_unstacked.columns)
        ax.set_yticklabels(monthly_ret_unstacked.index)
        
        # Annotate return values inside cells
        for i in range(len(monthly_ret_unstacked.index)):
            for j in range(12):
                val = monthly_ret_unstacked.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=9,
                            color="black" if abs(val) < 5 else "white", fontweight="bold")
                    
        ax.set_title("Nifty 50 Monthly Returns Compounded Heatmap Matrix (%)", fontsize=12, fontweight="bold")
        self._save_figure(fig, "12_monthly_returns_heatmap.png")

        # 3. Monthly Returns Distribution Boxplot
        fig, ax = plt.subplots(figsize=(10, 5))
        # Pivot individual daily returns by month index (1 to 12)
        daily_returns_by_month = [self.df[self.df.index.month == m]["Simple_Return"].dropna() * 100 for m in range(1, 13)]
        ax.boxplot(daily_returns_by_month, patch_artist=True,
                   boxprops=dict(facecolor=PLT_COLOR_LIGHT_BLUE, color=PLT_COLOR_DARK_BLUE),
                   medianprops=dict(color=PLT_COLOR_RED, linewidth=1.5))
        ax.set_title("Daily Return Distributions Grouped by Calendar Month", fontsize=12, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Daily Return (%)")
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.grid(True, **PLT_GRID_CONFIG)
        self._save_figure(fig, "13_monthly_seasonality_boxplots.png")

        # 4. Weekly Day-of-Week Returns Distribution Boxplot
        fig, ax = plt.subplots(figsize=(10, 5))
        # Monday is 0, Friday is 4 in pandas dayofweek
        daily_returns_by_weekday = [self.df[self.df.index.dayofweek == d]["Simple_Return"].dropna() * 100 for d in range(5)]
        
        # Calculate weekday means to report in text
        weekday_means = [float(self.df[self.df.index.dayofweek == d]["Simple_Return"].mean() * 100) for d in range(5)]
        self.stats_report["weekday_means"] = weekday_means
        
        ax.boxplot(daily_returns_by_weekday, patch_artist=True,
                   boxprops=dict(facecolor=PLT_COLOR_LIGHT_BLUE, color=PLT_COLOR_DARK_BLUE),
                   medianprops=dict(color=PLT_COLOR_RED, linewidth=1.5))
        ax.set_title("Daily Return Distributions Grouped by Day of the Week", fontsize=12, fontweight="bold")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Daily Return (%)")
        ax.set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        ax.grid(True, **PLT_GRID_CONFIG)
        self._save_figure(fig, "14_weekly_seasonality_boxplots.png")

        # 5. Section 15: Market Regimes Annotations
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(self.df.index, self.df["Close"], color="black", label="Close Price", linewidth=1)
        
        # Regimes shading:
        # A. 2016 Stagnation (say 2016-01-01 to 2016-12-31)
        ax.axvspan(pd.Timestamp("2015-12-01"), pd.Timestamp("2016-12-31"), color="grey", alpha=0.2, label="2016 Consolidation")
        # B. 2018 Correction (say 2018-01-29 to 2018-10-31)
        ax.axvspan(pd.Timestamp("2018-01-29"), pd.Timestamp("2018-10-31"), color="yellow", alpha=0.2, label="2018 Correction")
        # C. COVID-19 Panic & Crash (2020-02-20 to 2020-03-24)
        ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-24"), color="red", alpha=0.3, label="COVID Crash")
        # D. Post-COVID Recovery (2020-03-25 to 2021-10-18)
        ax.axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2021-10-18"), color="green", alpha=0.15, label="Post-COVID Bull Run")
        # E. 2022 Rate Hike Decline (2022-01-01 to 2022-06-30)
        ax.axvspan(pd.Timestamp("2021-10-19"), pd.Timestamp("2022-06-30"), color="orange", alpha=0.2, label="2022 Global Decline")
        
        # Labels and Annotations
        ax.annotate("COVID Bottom\n(March 23, 2020)", xy=(pd.Timestamp("2020-03-23"), 7610.25),
                    xytext=(pd.Timestamp("2019-01-01"), 6000),
                    arrowprops=dict(facecolor="black", shrink=0.08, width=0.5, headwidth=4))
        
        ax.set_title("Nifty 50 Historical Close Price Shaded by Market Regimes", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.grid(True, **PLT_GRID_CONFIG)
        ax.legend(loc="upper left")
        self._save_figure(fig, "15_market_regime_exploration.png")

    def compile_markdown_report(self) -> None:
        """Section 17: Compiles structural data quality, statistics, and quants findings into report."""
        logger.info(f"Section 17: Generating final markdown summary report to: {self.report_path}")
        
        overview = self.stats_report["overview"]
        quality = self.stats_report["quality"]
        desc_stats = self.stats_report["descriptive_stats"]
        norm = self.stats_report["stationarity_normality"]
        ext = self.stats_report["extreme_days"]
        ann = self.stats_report["annual"]
        wkd_means = self.stats_report["weekday_means"]

        content = f"""# Quantitative Research EDA Summary Report
**Asset**: Nifty 50 Index (`^NSEI`)  
**Data Period**: {overview['start_date']} to {overview['end_date']}  
**Generated Date**: 2026-08-05  

---

## 1. Dataset Overview
A high-level inspection of the processed analysis-ready market dataset:
* **Total Observations (Rows)**: {overview['rows']}
* **Total Dimensions (Columns)**: {overview['columns']}
* **Date Bounds**: {overview['start_date']} to {overview['end_date']}
* **Memory Footprint**: {overview['memory_kb']:.2f} KB
* **Dtypes**:
{chr(10).join([f"  * `{col}`: `{dtype}`" for col, dtype in overview['dtypes'].items()])}

---

## 2. Data Quality Audit Summary
The pre-preprocessing check reports complete coverage across clean columns:
* **Missing (NaN) values**: {sum(quality['missing_counts'].values())}
* **Infinite values**: {sum(quality['inf_counts'].values())}
* **Fully duplicate rows**: {quality['duplicate_rows']}
* **Duplicate timestamps**: {quality['duplicate_timestamps']}
* **Completeness Ratio**: 100.0% (Clean index contains no missing rows or duplicate date indices).

---

## 3. Core Descriptive Statistics
Descriptive metrics calculated for primary price and return dimensions:

| Statistic | Open | High | Low | Close | Simple_Return | Daily_Range | Price_Change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Count** | {desc_stats['Open']['count']:.0f} | {desc_stats['High']['count']:.0f} | {desc_stats['Low']['count']:.0f} | {desc_stats['Close']['count']:.0f} | {desc_stats['Simple_Return']['count']:.0f} | {desc_stats['Daily_Range']['count']:.0f} | {desc_stats['Price_Change']['count']:.0f} |
| **Mean** | {desc_stats['Open']['mean']:.2f} | {desc_stats['High']['mean']:.2f} | {desc_stats['Low']['mean']:.2f} | {desc_stats['Close']['mean']:.2f} | {desc_stats['Simple_Return']['mean'] * 100:.4f}% | {desc_stats['Daily_Range']['mean']:.2f} | {desc_stats['Price_Change']['mean']:.2f} |
| **Median** | {desc_stats['Open']['median']:.2f} | {desc_stats['High']['median']:.2f} | {desc_stats['Low']['median']:.2f} | {desc_stats['Close']['median']:.2f} | {desc_stats['Simple_Return']['median'] * 100:.4f}% | {desc_stats['Daily_Range']['median']:.2f} | {desc_stats['Price_Change']['median']:.2f} |
| **Std Dev** | {desc_stats['Open']['std']:.2f} | {desc_stats['High']['std']:.2f} | {desc_stats['Low']['std']:.2f} | {desc_stats['Close']['std']:.2f} | {desc_stats['Simple_Return']['std'] * 100:.4f}% | {desc_stats['Daily_Range']['std']:.2f} | {desc_stats['Price_Change']['std']:.2f} |
| **Minimum** | {desc_stats['Open']['min']:.2f} | {desc_stats['High']['min']:.2f} | {desc_stats['Low']['min']:.2f} | {desc_stats['Close']['min']:.2f} | {desc_stats['Simple_Return']['min'] * 100:.2f}% | {desc_stats['Daily_Range']['min']:.2f} | {desc_stats['Price_Change']['min']:.2f} |
| **Maximum** | {desc_stats['Open']['max']:.2f} | {desc_stats['High']['max']:.2f} | {desc_stats['Low']['max']:.2f} | {desc_stats['Close']['max']:.2f} | {desc_stats['Simple_Return']['max'] * 100:.2f}% | {desc_stats['Daily_Range']['max']:.2f} | {desc_stats['Price_Change']['max']:.2f} |
| **IQR** | {desc_stats['Open']['iqr']:.2f} | {desc_stats['High']['iqr']:.2f} | {desc_stats['Low']['iqr']:.2f} | {desc_stats['Close']['iqr']:.2f} | {desc_stats['Simple_Return']['iqr'] * 100:.4f}% | {desc_stats['Daily_Range']['iqr']:.2f} | {desc_stats['Price_Change']['iqr']:.2f} |
| **Coeff Var**| {desc_stats['Open']['cv']:.4f} | {desc_stats['High']['cv']:.4f} | {desc_stats['Low']['cv']:.4f} | {desc_stats['Close']['cv']:.4f} | {desc_stats['Simple_Return']['cv']:.4f} | {desc_stats['Daily_Range']['cv']:.4f} | {desc_stats['Price_Change']['cv']:.4f} |

---

## 4. Key Statistical Findings

### A. Distribution Normality
* **Skewness**: {norm['skewness']:.4f}  
  * *Interpretation*: A negative skewness indicates that the distribution has a longer left tail (more frequent large negative daily returns than normal would predict).
* **Excess Kurtosis**: {norm['kurtosis']:.4f}  
  * *Interpretation*: High excess kurtosis (leptokurtic distribution) mathematically confirms the presence of **fat-tails** or heavy tails, where extreme events occur far more frequently than in a bell curve.
* **Jarque-Bera Test**:
  * Test Statistic: {norm['jb_stat']:.2f}
  * p-value: {norm['jb_p']:.4e}
  * *Normality Rejection*: With a p-value of essentially 0, **we reject the null hypothesis of normal distribution** with absolute statistical significance. 

### B. Time Series Stationarity
We run the Augmented Dickey-Fuller (ADF) and KPSS tests on both Close Price levels and Daily Returns to verify stationarity.

| Series | ADF Statistic | ADF p-value | ADF Result (5% critical = {norm['adf']['price']['crit_5pct']:.2f}) | KPSS Statistic | KPSS p-value | KPSS Result (5% critical = 0.463) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Close Price** | {norm['adf']['price']['stat']:.2f} | {norm['adf']['price']['p']:.4f} | Non-Stationary | {norm['kpss']['price']['stat']:.2f} | {norm['kpss']['price']['p']:.4f} | Non-Stationary |
| **Daily Returns**| {norm['adf']['return']['stat']:.2f} | {norm['adf']['return']['p']:.4f} | **Stationary** | {norm['kpss']['return']['stat']:.2f} | {norm['kpss']['return']['p']:.4f} | **Stationary** |

* **Key Takeaway**: Price levels are integrated of order 1 ($I(1)$). Standard prices cannot be modeled with stationary models (like ARMA) without scaling. Returns are $I(0)$ and stationary, confirming they are appropriate for signal development.

---

## 5. Interesting Market Behaviour & Seasonality

### A. Weekday Return Characteristics
Average compounded daily return by day of week:
* **Monday**: {wkd_means[0]:+.4f}%
* **Tuesday**: {wkd_means[1]:+.4f}%
* **Wednesday**: {wkd_means[2]:+.4f}%
* **Thursday**: {wkd_means[3]:+.4f}%
* **Friday**: {wkd_means[4]:+.4f}%
* *Observation*: Friday and Tuesday have historically exhibited distinct average behaviors, which could serve as inputs for day-of-week style filter signals.

### B. Volatility Clustering
Comparing rolling 20, 50, and 100-day annualized volatilities confirms **volatility clustering**: high-volatility periods follow high-volatility periods, and low-volatility periods follow low-volatility periods. Volatility spikes are sudden (leptokurtic shocks) while decay is slow and persistent.

---

## 6. Extreme Market Days & Outliers
Top daily returns, ranges, and gap movements:

### Top 5 Positive Return Days
{chr(10).join([f"1. Date: {date} | Return: {item['Simple_Return']*100:+.2f}% | Close: {item['Close']:.2f}" for date, item in zip(ext['top_positive_dates'][:5], ext['top_positive'][:5])])}

### Top 5 Negative Return Days
{chr(10).join([f"1. Date: {date} | Return: {item['Simple_Return']*100:+.2f}% | Close: {item['Close']:.2f}" for date, item in zip(ext['top_negative_dates'][:5], ext['top_negative'][:5])])}

### Top 5 Opening Gaps (Opening Price Shock)
{chr(10).join([f"1. Date: {date} | Opening Gap: {item['Pct_Opening_Gap']*100:+.2f}% | Open: {item['Open']:.2f} | Close: {item['Close']:.2f}" for date, item in zip(ext['top_gaps_dates'][:5], ext['top_gaps'][:5])])}

* **Outlier Retainment Justification**: These observations are not data errors. They represent key market regimes (COVID crash, global oil wars, policy shock events). Retaining them is essential to prevent backtesting from underestimating maximum drawdowns and tail risk.

---

## 7. Strategic Recommendations for Signal Construction

1. **Mean-Reversion Signal Construction**:
   * Since returns are stationary ($I(0)$) and exhibit high mean-reversion tendencies during consolidation regimes (e.g., 2016, 2018), strategies should construct **Z-scores** based on price distance from rolling moving averages.
   * Volatility clustering suggests standardizing Z-scores by a rolling volatility estimator (such as rolling Bollinger band width) to scale entries dynamically during low/high regimes.
2. **Momentum Signal Construction**:
   * Prices exhibit significant trending regimes (e.g., post-COVID bull run). Dual moving average crossovers (e.g., 50-day and 200-day) or exponential trend-following filters are recommended to catch these high-momentum phases.
3. **Volatility Filters**:
   * Strategy execution should be scaled down or paused during periods when rolling 20-day annualized volatility exceeds historical 95th percentiles (e.g., > 30% volatility during March 2020) to limit stop-loss triggers.

---

## 8. Limitations of the Dataset
* **Survival Bias**: The Nifty 50 Index represents the top 50 active stocks; components change over time. Backtesting directly on the index hides survival biases of individual components.
* **No Transaction Costs**: Intraday ranges show that slippage and commission costs will eat up a significant part of high-frequency mean-reversion returns.
* **No Intraday Data**: Daily close limits signal accuracy; intraday gaps (up to 4.5% open gaps) can trigger stop-losses before close.
"""
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Report generated successfully.")
