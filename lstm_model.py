"""
LSTM Model for Energy Consumption Forecasting
Deep learning approach for time series prediction
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def create_sequences(data, seq_length):
    """
    Create sequences for LSTM training
    
    Parameters:
    -----------
    data : np.array
        Time series data
    seq_length : int
        Length of input sequences
        
    Returns:
    --------
    X, y : tuple
        Sequences and corresponding targets
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """
    Build LSTM model architecture
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (seq_length, features)
        
    Returns:
    --------
    model : Sequential
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_lstm_model(df, target_col, seq_length=24, epochs=10, batch_size=32):
    """
    Train LSTM model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data
    target_col : str
        Target column name
    seq_length : int
        Length of input sequences
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    model : Sequential
        Trained LSTM model
    scaler : MinMaxScaler
        Fitted scaler for data normalization
    """
    # Extract target variable
    data = df[target_col].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_length)
    
    # Split into train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train model
    model = build_lstm_model((seq_length, 1))
    print(f"Training LSTM model for {epochs} epochs...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_test, y_test), verbose=1)
    
    return model, scaler


def save_lstm_model(model, scaler, model_path='lstm_model.keras', scaler_path='scaler.pkl'):
    """
    Save trained LSTM model and scaler
    
    Parameters:
    -----------
    model : Sequential
        Trained LSTM model
    scaler : MinMaxScaler
        Fitted scaler
    model_path : str
        Path to save model
    scaler_path : str
        Path to save scaler
    """
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def main():
    """
    Main function to train LSTM model
    """
    import os
    
    # Configuration
    ENGINEERED_UCI = 'engineered_uci.csv'
    ENGINEERED_KAGGLE = 'engineered_kaggle.csv'
    PROCESSED_UCI = 'processed_uci_hourly.csv'
    PROCESSED_KAGGLE = 'processed_kaggle_hourly.csv'
    
    # Use engineered data if available, otherwise use processed data
    if os.path.exists(ENGINEERED_UCI):
        print("Using engineered UCI data...")
        df = pd.read_csv(ENGINEERED_UCI, parse_dates=['datetime'], index_col='datetime')
        target_col = 'Global_active_power'
    elif os.path.exists(ENGINEERED_KAGGLE):
        print("Using engineered Kaggle data...")
        df = pd.read_csv(ENGINEERED_KAGGLE, parse_dates=['datetime'], index_col='datetime')
        target_col = df.columns[0]
    elif os.path.exists(PROCESSED_UCI):
        print("Using processed UCI data (no engineered features)...")
        df = pd.read_csv(PROCESSED_UCI, parse_dates=['datetime'], index_col='datetime')
        target_col = 'Global_active_power'
    elif os.path.exists(PROCESSED_KAGGLE):
        print("Using processed Kaggle data (no engineered features)...")
        df = pd.read_csv(PROCESSED_KAGGLE, parse_dates=['datetime'], index_col='datetime')
        target_col = df.columns[0]
    else:
        print("No data found. Run data_preprocessing.py first.")
        return
    
    print(f"Data shape: {df.shape}")
    print(f"Target column: {target_col}")
    
    # Train model
    model, scaler = train_lstm_model(df, target_col, seq_length=24, epochs=10)
    
    # Save model
    save_lstm_model(model, scaler)
    
    print("\nLSTM model training complete!")


if __name__ == "__main__":
    main()
