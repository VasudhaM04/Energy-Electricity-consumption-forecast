"""
Data Loading and Preprocessing for Energy Consumption Forecasting
Supports UCI Household Power Consumption Dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_uci_data(filepath):
    """
    Load UCI Household Power Consumption Dataset
    
    Parameters:
    -----------
    filepath : str
        Path to household_power_consumption.txt
    
    Returns:
    --------
    pd.DataFrame
        Raw data with datetime index
    """
    print("Loading UCI Household Power Consumption Dataset...")
    
    # UCI dataset uses ';' as separator and '?' for missing values
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values='?',
        low_memory=False
    )
    
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    return df


def preprocess_uci_data(df):
    """
    Preprocess UCI dataset:
    - Merge date and time columns
    - Create datetime index
    - Handle missing values
    - Resample from 1-minute to hourly
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw UCI data
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed hourly data
    """
    print("\nPreprocessing data...")
    
    # Merge date and time columns
    df['datetime'] = df['Date'] + ' ' + df['Time']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Drop original Date and Time columns
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Handle missing values - forward fill then backward fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    print(f"Data after filling missing values: {df.shape}")
    
    # Resample from 1-minute to hourly (using mean)
    df_hourly = df.resample('h').mean()
    
    # Remove any remaining NaN values after resampling
    df_hourly.dropna(inplace=True)
    
    print(f"Data after hourly resampling: {df_hourly.shape}")
    print(f"Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
    print(f"Missing values after resampling: {df_hourly.isnull().sum().sum()}")
    
    return df_hourly


def load_kaggle_data(filepath):
    """
    Load and preprocess Kaggle dataset
    
    Parameters:
    -----------
    filepath : str
        Path to Kaggle CSV file
        
    Returns:
    --------
    pd.DataFrame
        Hourly data with datetime index
    """
    print(f"Loading Kaggle dataset from {filepath}...")
    
    df = pd.read_csv(filepath)
    
    # Find datetime column (case-insensitive)
    datetime_col = None
    for col in df.columns:
        if 'datetime' in col.lower():
            datetime_col = col
            break
    
    if datetime_col is None:
        raise ValueError(f"No datetime column found. Columns: {df.columns.tolist()}")
    
    # Convert datetime column
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def save_processed_data(df, output_path):
    """
    Save processed data to CSV
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed data
    output_path : str
        Output file path
    """
    df.to_csv(output_path)
    print(f"\nProcessed data saved to: {output_path}")


def main():
    """
    Main function to load and preprocess data
    """
    # Configuration
    UCI_DATA_PATH = 'household_power_consumption.txt'
    KAGGLE_DATA_PATH = 'PJME_hourly.csv'  # or any other Kaggle file
    OUTPUT_UCI = 'processed_uci_hourly.csv'
    OUTPUT_KAGGLE = 'processed_kaggle_hourly.csv'
    
    # Process UCI Dataset
    if os.path.exists(UCI_DATA_PATH):
        print("=" * 60)
        print("PROCESSING UCI DATASET")
        print("=" * 60)
        df_uci = load_uci_data(UCI_DATA_PATH)
        df_uci_processed = preprocess_uci_data(df_uci)
        save_processed_data(df_uci_processed, OUTPUT_UCI)
    
    # Process Kaggle Dataset
    if os.path.exists(KAGGLE_DATA_PATH):
        print("\n" + "=" * 60)
        print("PROCESSING KAGGLE DATASET")
        print("=" * 60)
        df_kaggle = load_kaggle_data(KAGGLE_DATA_PATH)
        save_processed_data(df_kaggle, OUTPUT_KAGGLE)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
