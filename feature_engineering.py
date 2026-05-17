"""
Feature Engineering for Energy Consumption Forecasting
Creates temporal, lag, rolling, and interaction features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import holidays


def create_temporal_features(df):
    """
    Create temporal features from datetime index
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    
    Returns:
    --------
    pd.DataFrame
        Data with temporal features added
    """
    df = df.copy()
    
    # Basic temporal features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Cyclical features (for capturing periodic patterns)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Holiday feature (using US holidays as default)
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.map(lambda x: int(x.date() in us_holidays))
    
    # Time of day categories
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    
    print("Temporal features created")
    return df


def create_lag_features(df, target_column, lags=[1, 2, 3, 6, 12, 24, 48, 168]):
    """
    Create lag features for target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    target_column : str
        Column to create lags for
    lags : list
        List of lag periods (in hours)
    
    Returns:
    --------
    pd.DataFrame
        Data with lag features added
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_column}_lag_{lag}h'] = df[target_column].shift(lag)
    
    print(f"Lag features created for lags: {lags}")
    return df


def create_rolling_features(df, target_column, windows=[6, 12, 24, 48, 168]):
    """
    Create rolling window statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    target_column : str
        Column to create rolling features for
    windows : list
        List of window sizes (in hours)
    
    Returns:
    --------
    pd.DataFrame
        Data with rolling features added
    """
    df = df.copy()
    
    for window in windows:
        df[f'{target_column}_rolling_mean_{window}h'] = df[target_column].rolling(window=window).mean()
        df[f'{target_column}_rolling_std_{window}h'] = df[target_column].rolling(window=window).std()
        df[f'{target_column}_rolling_max_{window}h'] = df[target_column].rolling(window=window).max()
        df[f'{target_column}_rolling_min_{window}h'] = df[target_column].rolling(window=window).min()
    
    print(f"Rolling features created for windows: {windows}")
    return df


def create_interaction_features(df):
    """
    Create interaction features between variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with features
    
    Returns:
    --------
    pd.DataFrame
        Data with interaction features added
    """
    df = df.copy()
    
    # Temperature-hour interaction (if temperature column exists)
    if 'Temperature' in df.columns or 'temp' in df.columns:
        temp_col = 'Temperature' if 'Temperature' in df.columns else 'temp'
        df[f'{temp_col}_x_hour'] = df[temp_col] * df['hour']
        df[f'{temp_col}_x_is_weekend'] = df[temp_col] * df['is_weekend']
    
    # Consumption-hour interaction (if consumption column exists)
    if 'Global_active_power' in df.columns:
        df['consumption_x_hour'] = df['Global_active_power'] * df['hour']
    
    print("Interaction features created")
    return df


def create_diff_features(df, target_column, periods=[1, 24]):
    """
    Create difference features (change from previous periods)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    target_column : str
        Column to create differences for
    periods : list
        List of periods to difference
    
    Returns:
    --------
    pd.DataFrame
        Data with difference features added
    """
    df = df.copy()
    
    for period in periods:
        df[f'{target_column}_diff_{period}h'] = df[target_column].diff(periods=period)
    
    print(f"Difference features created for periods: {periods}")
    return df


def engineer_features(df, target_column='Global_active_power'):
    """
    Complete feature engineering pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    target_column : str
        Target variable column name
    
    Returns:
    --------
    pd.DataFrame
        Data with all engineered features
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print(f"Original shape: {df.shape}")
    
    # Step 1: Temporal features
    df = create_temporal_features(df)
    print(f"After temporal features: {df.shape}")
    
    # Step 2: Lag features
    df = create_lag_features(df, target_column, lags=[1, 2, 3, 6, 12, 24, 48, 168])
    print(f"After lag features: {df.shape}")
    
    # Step 3: Rolling features
    df = create_rolling_features(df, target_column, windows=[6, 12, 24, 48, 168])
    print(f"After rolling features: {df.shape}")
    
    # Step 4: Difference features
    df = create_diff_features(df, target_column, periods=[1, 24])
    print(f"After difference features: {df.shape}")
    
    # Step 5: Interaction features
    df = create_interaction_features(df)
    print(f"After interaction features: {df.shape}")
    
    # Remove rows with NaN (created by lag/rolling features)
    df.dropna(inplace=True)
    print(f"After removing NaN: {df.shape}")
    
    print("=" * 60)
    print(f"Final shape: {df.shape}")
    print(f"Total features: {df.shape[1]}")
    print("=" * 60)
    
    return df


def save_engineered_data(df, output_path):
    """
    Save engineered data to CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        Engineered data
    output_path : str
        Output file path
    """
    df.to_csv(output_path)
    print(f"\nEngineered data saved to: {output_path}")


def main():
    """
    Main function to run feature engineering
    """
    import os
    
    # Configuration
    PROCESSED_UCI = 'processed_uci_hourly.csv'
    PROCESSED_KAGGLE = 'processed_kaggle_hourly.csv'
    OUTPUT_UCI = 'engineered_uci.csv'
    OUTPUT_KAGGLE = 'engineered_kaggle.csv'
    
    # Use UCI dataset if available, otherwise Kaggle
    if os.path.exists(PROCESSED_UCI):
        print("Using UCI dataset for feature engineering...")
        df = pd.read_csv(PROCESSED_UCI, parse_dates=['datetime'], index_col='datetime')
        target_col = 'Global_active_power'
        df_engineered = engineer_features(df, target_col)
        save_engineered_data(df_engineered, OUTPUT_UCI)
        
    elif os.path.exists(PROCESSED_KAGGLE):
        print("Using Kaggle dataset for feature engineering...")
        df = pd.read_csv(PROCESSED_KAGGLE, parse_dates=['datetime'], index_col='datetime')
        target_col = df.columns[0]  # First column is usually the consumption
        df_engineered = engineer_features(df, target_col)
        save_engineered_data(df_engineered, OUTPUT_KAGGLE)
        
    else:
        print("No processed data found. Run DataPreprocessing/data_preprocessing.py first.")
        return
    
    print("\nFeature Engineering Complete!")


if __name__ == "__main__":
    main()
