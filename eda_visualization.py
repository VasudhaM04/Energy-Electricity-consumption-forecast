"""
Exploratory Data Analysis and Visualization for Energy Consumption
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_processed_data(filepath):
    """Load processed data"""
    df = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    print(f"Loaded data: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df


def plot_time_series(df, column='Global_active_power', title='Energy Consumption Over Time'):
    """
    Plot time series of energy consumption
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    column : str
        Column to plot
    title : str
        Plot title
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode='lines',
        name=column,
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Energy (kW)',
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig


def plot_seasonal_decomposition(df, column='Global_active_power'):
    """
    Plot seasonal decomposition
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    column : str
        Column to decompose
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Resample to daily for clearer decomposition
    df_daily = df[column].resample('D').mean().dropna()
    
    # Perform decomposition
    decomposition = seasonal_decompose(df_daily, model='additive', period=7)
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
        vertical_spacing=0.08
    )
    
    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed,
                             mode='lines', name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend,
                             mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal,
                             mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid,
                             mode='lines', name='Residual'), row=4, col=1)
    
    fig.update_layout(height=900, template='plotly_dark', showlegend=False)
    
    return fig


def plot_hourly_patterns(df, column='Global_active_power'):
    """
    Plot average consumption by hour of day
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    column : str
        Column to analyze
    """
    # Extract hour
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    
    # Calculate mean by hour
    hourly_mean = df_copy.groupby('hour')[column].mean()
    
    fig = go.Figure(data=go.Bar(
        x=hourly_mean.index,
        y=hourly_mean.values,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Average Energy Consumption by Hour of Day',
        xaxis_title='Hour (0-23)',
        yaxis_title='Average Consumption (kW)',
        template='plotly_dark'
    )
    
    return fig


def plot_daily_patterns(df, column='Global_active_power'):
    """
    Plot average consumption by day of week
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    column : str
        Column to analyze
    """
    df_copy = df.copy()
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['day_name'] = df_copy.index.day_name()
    
    daily_mean = df_copy.groupby('day_name')[column].mean()
    
    # Reorder by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_mean = daily_mean.reindex(day_order)
    
    fig = go.Figure(data=go.Bar(
        x=daily_mean.index,
        y=daily_mean.values,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Average Energy Consumption by Day of Week',
        xaxis_title='Day',
        yaxis_title='Average Consumption (kW)',
        template='plotly_dark'
    )
    
    return fig


def plot_monthly_patterns(df, column='Global_active_power'):
    """
    Plot average consumption by month
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    column : str
        Column to analyze
    """
    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month
    df_copy['month_name'] = df_copy.index.month_name()
    
    monthly_mean = df_copy.groupby('month_name')[column].mean()
    
    # Reorder by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_mean = monthly_mean.reindex(month_order)
    
    fig = go.Figure(data=go.Bar(
        x=monthly_mean.index,
        y=monthly_mean.values,
        marker_color='orange'
    ))
    
    fig.update_layout(
        title='Average Energy Consumption by Month',
        xaxis_title='Month',
        yaxis_title='Average Consumption (kW)',
        template='plotly_dark'
    )
    
    return fig


def plot_heatmap_hour_day(df, column='Global_active_power'):
    """
    Plot heatmap of consumption by hour and day of week
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    column : str
        Column to analyze
    """
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['day_of_week'] = df_copy.index.dayofweek
    
    # Create pivot table
    pivot_table = df_copy.pivot_table(
        values=column,
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )
    
    # Rename columns
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table.columns = day_names
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis',
        colorbar=dict(title='Consumption (kW)')
    ))
    
    fig.update_layout(
        title='Energy Consumption Heatmap (Hour vs Day of Week)',
        xaxis_title='Day of Week',
        yaxis_title='Hour of Day',
        template='plotly_dark'
    )
    
    return fig


def plot_correlation_matrix(df):
    """
    Plot correlation matrix of features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with features
    """
    # Calculate correlation
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        template='plotly_dark',
        width=800,
        height=800
    )
    
    return fig


def plot_distribution(df, column='Global_active_power'):
    """
    Plot distribution of energy consumption
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    column : str
        Column to analyze
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Histogram', 'Box Plot'])
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=50,
        name='Distribution',
        marker_color='lightblue'
    ), row=1, col=1)
    
    # Box plot
    fig.add_trace(go.Box(
        y=df[column],
        name='Box Plot',
        marker_color='lightgreen'
    ), row=1, col=2)
    
    fig.update_layout(
        title=f'Distribution of {column}',
        template='plotly_dark',
        showlegend=False
    )
    
    return fig


def generate_all_visualizations(df, output_dir='visualizations'):
    """
    Generate all visualizations and save to HTML files
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed data
    output_dir : str
        Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the main column
    if 'Global_active_power' in df.columns:
        main_col = 'Global_active_power'
    elif df.shape[1] == 1:
        main_col = df.columns[0]
    else:
        main_col = df.columns[0]
    
    print(f"Using column: {main_col}")
    
    # Generate visualizations
    print("Generating time series plot...")
    fig1 = plot_time_series(df, main_col)
    fig1.write_html(f"{output_dir}/01_time_series.html")
    
    print("Generating seasonal decomposition...")
    fig2 = plot_seasonal_decomposition(df, main_col)
    fig2.write_html(f"{output_dir}/02_seasonal_decomposition.html")
    
    print("Generating hourly patterns...")
    fig3 = plot_hourly_patterns(df, main_col)
    fig3.write_html(f"{output_dir}/03_hourly_patterns.html")
    
    print("Generating daily patterns...")
    fig4 = plot_daily_patterns(df, main_col)
    fig4.write_html(f"{output_dir}/04_daily_patterns.html")
    
    print("Generating monthly patterns...")
    fig5 = plot_monthly_patterns(df, main_col)
    fig5.write_html(f"{output_dir}/05_monthly_patterns.html")
    
    print("Generating heatmap...")
    fig6 = plot_heatmap_hour_day(df, main_col)
    fig6.write_html(f"{output_dir}/06_heatmap.html")
    
    print("Generating correlation matrix...")
    fig7 = plot_correlation_matrix(df)
    fig7.write_html(f"{output_dir}/07_correlation_matrix.html")
    
    print("Generating distribution plot...")
    fig8 = plot_distribution(df, main_col)
    fig8.write_html(f"{output_dir}/08_distribution.html")
    
    print(f"\nAll visualizations saved to: {output_dir}/")


def main():
    """
    Main function to run EDA
    """
    # Configuration
    PROCESSED_UCI = 'processed_uci_hourly.csv'
    PROCESSED_KAGGLE = 'processed_kaggle_hourly.csv'
    
    # Use UCI dataset if available, otherwise Kaggle
    if os.path.exists(PROCESSED_UCI):
        print("Using UCI dataset for EDA...")
        df = load_processed_data(PROCESSED_UCI)
    elif os.path.exists(PROCESSED_KAGGLE):
        print("Using Kaggle dataset for EDA...")
        df = load_processed_data(PROCESSED_KAGGLE)
    else:
        print("No processed data found. Run data_preprocessing.py first.")
        return
    
    # Generate all visualizations
    generate_all_visualizations(df)
    
    print("\nEDA Complete!")


if __name__ == "__main__":
    main()
