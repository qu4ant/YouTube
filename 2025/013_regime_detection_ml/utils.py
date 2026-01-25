"""
Utility functions for regime detection analysis
"""
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, DatetimeTickFormatter, Span, Label
from bokeh.palettes import Category20, Turbo256
from bokeh.io import curdoc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def calculate_rolling_stats(series, window=24, prefix=''):
    """Calculer les statistiques rolling pour une série stationnaire"""
    rolling_stats = pd.DataFrame(index=series.index)

    # Statistiques de base avec période dans le nom (pour séries stationnaires)
    rolling_stats[f'{prefix}_{window}_std'] = series.rolling(window, min_periods=1).std()
    rolling_stats[f'{prefix}_{window}_kurt'] = series.rolling(window, min_periods=4).apply(
        lambda x: x.kurt() if len(x) >= 4 else 0)
    rolling_stats[f'{prefix}_{window}_skew'] = series.rolling(window, min_periods=3).apply(
        lambda x: x.skew() if len(x) >= 3 else 0)

    return rolling_stats

def calculate_sma_distance(series, window=24, prefix=''):
    """Calculer la distance à la SMA pour une série non-stationnaire (prix)"""
    sma = series.rolling(window, min_periods=1).mean()
    dist_sma = pd.DataFrame(index=series.index)
    dist_sma[f'{prefix}_{window}_dist_sma'] = (series - sma) / sma * 100
    return dist_sma

def visualize_split(train_size, test_size):
    total = train_size + test_size
    train_pct = int(train_size / total * 50)  # 50 chars wide
    test_pct = int(test_size / total * 50)

    # ANSI color codes
    BLUE = '\033[94m'
    ORANGE = '\033[38;5;208m'
    RESET = '\033[0m'

    print("Dataset Split Visualization")
    print("=" * 52)
    print(f"Train: {BLUE}{'█' * train_pct}{'░' * test_pct}{RESET} {train_size} samples ({train_size/total*100:.0f}%)")
    print(f"Test:  {ORANGE}{'░' * train_pct}{'█' * test_pct}{RESET} {test_size} samples ({test_size/total*100:.0f}%)")
    print("=" * 52)

def plot_regime_clusters(df, train_size=0.75, colormap='Set1', figsize=(1000, 600)):
    """
    Create an interactive Bokeh plot showing Bitcoin prices colored by regime clusters
    with train/test split visualization

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'close' prices, 'cluster' labels, and 'confidence' (optional)
    train_size : float
        Proportion of data used for training (default 0.75)
    colormap : str
        Matplotlib colormap name to use for cluster colors (default 'Set1')
        Popular options: 'Set1', 'Set2', 'Set3', 'Dark2', 'tab10', 'viridis', 'plasma', 'hsv'
    figsize : tuple
        Figure dimensions as (width, height) in pixels (default (1000, 600))

    Returns:
    --------
    bokeh.plotting.figure : The configured plot object
    """
    # Prepare data for visualization
    # Only copy the columns we need for plotting (avoids copying heavy autocorr columns)
    cols_to_copy = ['close', 'cluster']
    if 'confidence' in df.columns:
        cols_to_copy.append('confidence')

    viz_df = df[cols_to_copy].copy()
    viz_df = viz_df.dropna(subset=['cluster'])  # Remove any NaN clusters
    viz_df['cluster_str'] = viz_df['cluster'].astype(int).astype(str)  # Convert to string for categorical coloring

    viz_df = viz_df.reset_index()  # Reset to get timestamp as column

    # Calculate train/test split point
    split_index = int(len(viz_df) * train_size)
    split_timestamp = viz_df.iloc[split_index]['timestamp']
    print(f"🔍 Train/Test split at: {split_timestamp}")
    print(f"   Training data: {split_index:,} samples")
    print(f"   Test data: {len(viz_df) - split_index:,} samples")

    # Dynamically determine number of clusters and select appropriate palette
    n_clusters = viz_df['cluster_str'].nunique()
    print(f"📊 Number of unique clusters detected: {n_clusters}")

    # Generate colors using the specified matplotlib colormap
    cmap = plt.cm.get_cmap(colormap)
    colors = [mcolors.to_hex(cmap(i / max(n_clusters-1, 1))) for i in range(n_clusters)]

    # Map clusters to colors
    cluster_ids = sorted(viz_df['cluster_str'].unique())
    color_mapping = {str(cluster): colors[i] for i, cluster in enumerate(cluster_ids)}

    # Assign colors to dataframe
    viz_df['color'] = viz_df['cluster_str'].map(color_mapping)

    print(f"🎨 Color mapping: {color_mapping}")

    # Create figure with dark theme styling
    p = figure(
        width=figsize[0],
        height=figsize[1],
        title="Bitcoin Spot Price with Gaussian Mixture Regime Detection (Train/Test Split)",
        x_axis_type="datetime",
        x_axis_label="Time",
        y_axis_label="Spot Close Price (USDT)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above"
    )

    # Apply dark theme styling
    p.background_fill_color = "#1a1a1a"  # Dark background
    p.border_fill_color = "#1a1a1a"
    p.outline_line_color = "#333333"

    # Style the grid
    p.grid.grid_line_color = "#3a3a3a"
    p.grid.grid_line_alpha = 0.3

    # Style axes
    p.xaxis.axis_label_text_color = "#c0c0c0"
    p.yaxis.axis_label_text_color = "#c0c0c0"
    p.xaxis.major_label_text_color = "#a0a0a0"
    p.yaxis.major_label_text_color = "#a0a0a0"
    p.xaxis.axis_line_color = "#505050"
    p.yaxis.axis_line_color = "#505050"
    p.xaxis.major_tick_line_color = "#505050"
    p.yaxis.major_tick_line_color = "#505050"
    p.xaxis.minor_tick_line_color = "#404040"
    p.yaxis.minor_tick_line_color = "#404040"

    # Style title
    p.title.text_color = "#e0e0e0"
    p.title.text_font_size = "16pt"

    # Plot scatter points for each cluster
    for cluster_id in cluster_ids:
        cluster_data_df = viz_df[viz_df['cluster_str'] == cluster_id].copy()

        # Calculer la taille en fonction de la confidence si disponible
        if 'confidence' in viz_df.columns:
            # Mapper confidence [0, 1] vers taille [2, 10]
            cluster_data_df['size'] = cluster_data_df['confidence'] * 5
            size_param = 'size'
        else:
            size_param = 5  # Taille fixe par défaut

        # Convert to ColumnDataSource
        from bokeh.models import ColumnDataSource
        cluster_data = ColumnDataSource(cluster_data_df)

        p.scatter(
            x='timestamp',
            y='close',
            source=cluster_data,
            size=size_param,
            color=color_mapping[cluster_id],
            alpha=0.7,  # Opacité fixe
            legend_label=f"Cluster {cluster_id}",
            name=f"cluster_{cluster_id}"
        )

    # Add vertical line to separate train/test data
    train_test_line = Span(
        location=split_timestamp,
        dimension='height',
        line_color='white',
        line_dash='dashed',
        line_width=2,
        line_alpha=0.6
    )
    p.add_layout(train_test_line)

    # Add labels for train and test regions
    # Calculate positions for labels (middle of each region and near top of price range)
    train_label_x = viz_df.iloc[split_index // 2]['timestamp']
    test_label_x = viz_df.iloc[split_index + (len(viz_df) - split_index) // 2]['timestamp']
    label_y = viz_df['close'].max() * 0.95  # Position labels near top of chart

    # Add "Training Data" label
    train_label = Label(
        x=train_label_x,
        y=label_y,
        text='TRAINING DATA (75%)',
        text_color='#00D9FF',
        text_font_size='12pt',
        text_font_style='bold',
        text_alpha=0.7,
        x_units='data',
        y_units='data'
    )
    p.add_layout(train_label)

    # Add "Test Data" label
    test_label = Label(
        x=test_label_x,
        y=label_y,
        text='TEST DATA (25%)',
        text_color='#FF6B6B',
        text_font_size='12pt',
        text_font_style='bold',
        text_alpha=0.7,
        x_units='data',
        y_units='data'
    )
    p.add_layout(test_label)

    # Configure hover tool
    tooltips = [
        ("Date", "@timestamp{%Y-%m-%d %H:%M}"),
        ("Price", "@close{$0,0.00}"),
        ("Cluster", "@cluster_str")
    ]

    # Add confidence score if available
    if 'confidence' in viz_df.columns:
        tooltips.append(("Confidence", "@confidence{0.000}"))

    hover = HoverTool(tooltips=tooltips, formatters={'@timestamp': 'datetime'})

    p.add_tools(hover)

    # Style legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"  # Click legend to hide/show clusters
    p.legend.label_text_color = "#c0c0c0"
    p.legend.background_fill_color = "#2a2a2a"
    p.legend.background_fill_alpha = 0.8
    p.legend.border_line_color = "#505050"
    p.legend.border_line_alpha = 0.5

    # Format x-axis datetime
    p.xaxis.formatter = DatetimeTickFormatter(
        hours="%H:%M",
        days="%Y-%m-%d",
        months="%Y-%m",
        years="%Y"
    )

    # Ensure notebook output is configured for Jupyter environments
    try:
        # Check if we're in a notebook environment
        get_ipython()
        output_notebook()
        curdoc().theme = 'dark_minimal'
    except NameError:
        # Not in notebook, normal behavior
        pass

    # Show the plot
    show(p)

    return p

def plot_feature_grid(df, cols_per_row=3, figsize_per_plot=(5, 3),
                     line_color='cyan', text_color='white',
                     background_color=None, title_prefix='',
                     show_grid=False, tight_layout=True):
    """
    Create a grid of subplots for exploratory data analysis of features.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features to plot. Should have a MultiIndex with 'timestamp' level
        or a regular index that will be used as x-axis
    cols_per_row : int
        Number of columns per row in the subplot grid (default 3)
    figsize_per_plot : tuple
        Figure size per individual plot (width, height) in inches (default (5, 3))
    line_color : str
        Color for the plot lines (default 'cyan')
    text_color : str
        Color for text, labels, and spines (default 'white')
    background_color : str or None
        Background color for the figure (default None, which uses default background)
    title_prefix : str
        Optional prefix to add to the figure title (default '')
    show_grid : bool
        Whether to show gridlines on plots (default False)
    tight_layout : bool
        Whether to apply tight layout to prevent overlap (default True)

    Returns:
    --------
    fig, axes : tuple
        The matplotlib figure and axes objects for further customization

    Examples:
    ---------
    >>> # Basic usage
    >>> plot_feature_grid(features)

    >>> # Custom styling
    >>> fig, axes = plot_feature_grid(features, cols_per_row=4,
    ...                               line_color='green', text_color='yellow')

    >>> # With different dataset
    >>> plot_feature_grid(validation_features, title_prefix='Validation Set: ')
    """

    # Handle index - check if it's MultiIndex with timestamp level
    if isinstance(df.index, pd.MultiIndex):
        if 'timestamp' in df.index.names:
            timestamps = df.index.get_level_values('timestamp')
        else:
            # Use the first level if no timestamp level exists
            timestamps = df.index.get_level_values(0)
    else:
        # Use regular index
        timestamps = df.index

    # Calculate grid dimensions
    n_features = len(df.columns)
    n_rows = int(np.ceil(n_features / cols_per_row))

    # Calculate total figure size
    fig_width = figsize_per_plot[0] * cols_per_row
    fig_height = figsize_per_plot[1] * n_rows

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=cols_per_row,
                            figsize=(fig_width, fig_height))

    # Set background color if specified
    if background_color:
        fig.patch.set_facecolor(background_color)

    # Add title if prefix is provided
    if title_prefix:
        fig.suptitle(f'{title_prefix}Feature Analysis Grid',
                    fontsize=16, color=text_color, y=1.02)

    # Flatten axes array for easier iteration
    # Handle case where there's only one subplot
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Plot each feature
    for i, col in enumerate(df.columns):
        ax = axes[i]

        # Plot the data
        ax.plot(timestamps, df[col], color=line_color, linewidth=0.8)

        # Set title
        ax.set_title(col, color=text_color, fontsize=10, pad=5)

        # Style the axes
        ax.tick_params(colors=text_color, labelsize=8)

        # Set spine colors
        for spine in ['bottom', 'top', 'right', 'left']:
            ax.spines[spine].set_color(text_color)

        # Optional grid
        if show_grid:
            ax.grid(True, alpha=0.2, color=text_color, linestyle='--')

        # Set background color for each subplot if specified
        if background_color:
            ax.set_facecolor(background_color)

        # Rotate x-axis labels if they're dates
        if hasattr(timestamps, 'dtype') and np.issubdtype(timestamps.dtype, np.datetime64):
            ax.tick_params(axis='x', rotation=45)

        # Format y-axis for better readability
        ax.yaxis.get_major_formatter().set_scientific(False)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        if j < len(axes):
            axes[j].set_visible(False)

    # Apply tight layout if requested
    if tight_layout:
        plt.tight_layout()

    # Show the plot
    plt.show()

    return