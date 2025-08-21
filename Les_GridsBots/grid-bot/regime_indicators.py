import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from hurst import compute_Hc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller

def rolling_hurst(series, window):
    """Calculate rolling Hurst exponent using the hurst library"""
    # Convert to numpy array if it's a pandas series
    if isinstance(series, pd.Series):
        series = series.values
    
    # Create sliding windows
    windows = sliding_window_view(series, window)
    
    # Calculate Hurst for each window
    hurst_values = []
    for window_data in windows:
        try:
            H, _, _ = compute_Hc(window_data, kind='price')
            hurst_values.append(H)
        except:
            hurst_values.append(np.nan)
    
    hurst_values = np.array(hurst_values)
    
    # Pad the beginning with NaN to match original series length
    padding = np.full(window - 1, np.nan)
    return np.concatenate([padding, hurst_values])

def rolling_adf(series, window=63, regression='c'):
    """
    Calculate rolling Augmented Dickey-Fuller test statistic.
    
    Parameters:
    -----------
    series : pd.Series
        Price series to analyze
    window : int
        Rolling window size for ADF calculation
    regression : str
        Type of regression to use in ADF test:
        'c' : constant only (default)
        'ct' : constant and trend
        'nc' : no constant, no trend
    
    Returns:
    --------
    pd.Series with ADF test statistics
    """
    # Create empty series to store results
    adf_stats = pd.Series(index=series.index, dtype=float)
    
    # Loop through the series using a rolling window
    for i in range(window, len(series) + 1):
        # Get window of data
        window_data = series.iloc[i-window:i]
        try:
            # Run ADF test and store test statistic
            result = adfuller(window_data, regression=regression)
            adf_stats.iloc[i-1] = result[0]  # Test statistic
        except:
            adf_stats.iloc[i-1] = np.nan
    
    return adf_stats

def plot_regime_indicators(df, price_col='close', figsize=(25, 10), threshold=0.6):
    """
    Plot price and Hurst exponent with strong trend regime indicators
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the price data and indicators
    price_col : str, default='close'
        Name of the column containing price data
    figsize : tuple, default=(25, 10)
        Figure size in inches (width, height)
    threshold : float, default=0.6
        Threshold for strong trend identification
    """
    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1])
    gs.update(hspace=0)  # Remove spacing between subplots

    # Create mask for strong trend regime
    trend_mask = df['hurst'] > threshold

    # Plot price
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df[price_col], color='white', linewidth=1)
    ax1.set_title('Price and Strong Trend Regime Indicators')
    ax1.set_ylabel('Price')
    
    # Add trend regime background
    ax1.fill_between(df.index, *ax1.get_ylim(), where=trend_mask, color='red', alpha=0.3)
    ax1.grid(True)

    # Plot Hurst exponent
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Share x-axis with price plot
    ax2.plot(df.index, df['hurst'], color='skyblue', linewidth=1)
    
    # Add trend regime background
    ax2.fill_between(df.index, *ax2.get_ylim(), where=trend_mask, color='red', alpha=0.3)
    
    # Add reference line for strong trend threshold
    ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('Hurst Exponent')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    ax2.set_ylim(0, 1)  # Set y-limits for Hurst plot

    # Remove x-axis visibility from top plot
    ax1.xaxis.set_visible(False)

    # Adjust layout
    plt.tight_layout()
    return fig

def detect_high_vol(series, window=21, threshold=0.20):
    """
    Detect high volatility periods using rolling standard deviation.
    
    Parameters:
    -----------
    series : pd.Series
        Price series to analyze
    window : int
        Rolling window size for standard deviation calculation
    threshold : float
        Threshold for high volatility (annualized)
    
    Returns:
    --------
    pd.DataFrame with:
        - std: Rolling standard deviation (annualized)
        - is_high_vol: Boolean series (True when in high vol regime)
    """
    log_ret = np.log(series / series.shift(1))
    vol = log_ret.rolling(window=window).std() * np.sqrt(252)
    
    return pd.DataFrame({
        'std': vol,
        'is_high_vol': vol > threshold
    })

def plot_vol_regime(df, price_col='close', window=21, threshold=0.20, figsize=(15, 8)):
    """
    Plot price with high volatility periods highlighted.
    """
    regime = detect_high_vol(df[price_col], window, threshold)
    
    # Create figure with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1], sharex=True)
    
    # Price plot with high vol periods highlighted
    ax1.plot(df.index, df[price_col], 'white', label='Price')
    for i in range(len(df)):
        if regime['is_high_vol'].iloc[i]:
            ax1.axvspan(df.index[i], 
                       df.index[i+1] if i < len(df)-1 else df.index[-1],
                       alpha=0.2, color='red')
    ax1.set_title('Price with High Volatility Periods')
    ax1.grid(True)
    
    # Volatility plot
    ax2.plot(df.index, regime['std'], 'purple', label='Volatility')
    ax2.axhline(y=threshold, color='r', ls='--', label=f'Threshold ({threshold:.0%})')
    ax2.set_title('Rolling Volatility')
    ax2.grid(True)
    ax2.legend()
    
    # Remove space between subplots and adjust layout
    plt.subplots_adjust(hspace=0.1)
    
    # Only show x-axis labels on bottom subplot
    ax1.tick_params(labelbottom=False)
    
    return fig

def plot_adf_regime(df, price_col='close', window=63, critical_value=-2.87, figsize=(15, 8), regression='c', use_returns=False):
    """
    Plot price with non-mean-reverting periods based on ADF test highlighted in red.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    price_col : str
        Column name for price data
    window : int
        Rolling window size for ADF test
    critical_value : float
        ADF test critical value threshold (default -2.87 for 5% significance)
    figsize : tuple
        Figure size
    regression : str
        ADF test regression type ('c', 'ct', or 'nc')
    use_returns : bool, default=False
        If True, apply ADF test to log returns instead of raw prices
    
    Returns:
    --------
    fig : Figure object
    adf_stats : pd.Series
        Series of rolling ADF test statistics
    """
    # Calculate log returns if needed
    if use_returns:
        series = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        plot_title = 'Returns with Non-Stationary Periods Highlighted'
        stat_title = 'Rolling ADF Test Statistic (Log Returns)'
    else:
        series = df[price_col]
        plot_title = 'Price with Non-Mean-Reverting Periods Highlighted'
        stat_title = 'Rolling ADF Test Statistic (Price)'
    
    # Calculate rolling ADF statistics
    adf_stats = rolling_adf(series, window=window, regression=regression)
    
    # Create figure with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1], sharex=True)
    
    # Create non-mean-reverting mask (where ADF statistic is above critical value)
    non_mean_reverting = adf_stats >= critical_value
    
    # Plot price or returns
    if use_returns:
        # Plot returns
        ax1.plot(series.index, series, 'white', label='Log Returns')
    else:
        # Plot price
        ax1.plot(df.index, df[price_col], 'white', label='Price')
    
    # Highlight non-mean-reverting periods
    for i in range(len(series.index)):
        if i < len(non_mean_reverting) and non_mean_reverting.iloc[i]:
            ax1.axvspan(series.index[i], 
                       series.index[i+1] if i < len(series.index)-1 else series.index[-1],
                       alpha=0.2, color='red')
    
    ax1.set_title(plot_title)
    ax1.grid(True)
    
    # ADF test statistic plot
    ax2.plot(adf_stats.index, adf_stats, 'cyan', label='ADF Statistic')
    ax2.axhline(y=critical_value, color='g', ls='--', label=f'Critical Value ({critical_value})')
    ax2.set_title(stat_title)
    ax2.grid(True)
    ax2.legend()
    
    # Remove space between subplots and adjust layout
    plt.subplots_adjust(hspace=0.1)
    
    # Only show x-axis labels on bottom subplot
    ax1.tick_params(labelbottom=False)
    
    return fig, adf_stats
