"""
ARIMA Model for Energy Consumption Forecasting
Statistical approach for time series prediction
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def check_stationarity(series):
    """
    Check if time series is stationary using Augmented Dickey-Fuller test
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
        
    Returns:
    --------
    is_stationary : bool
        True if series is stationary
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value < 0.05


def train_arima_model(df, target_col, order=(1, 1, 1)):
    """
    Train ARIMA model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data
    target_col : str
        Target column name
    order : tuple
        ARIMA order (p, d, q)
        
    Returns:
    --------
    model : ARIMAResults
        Trained ARIMA model
    """
    # Extract target variable
    data = df[target_col]
    
    # Check stationarity
    is_stationary = check_stationarity(data)
    print(f"Series is {'stationary' if is_stationary else 'not stationary'}")
    
    # Train ARIMA model
    print(f"Training ARIMA model with order {order}...")
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    
    print(f"ARIMA model trained successfully")
    print(f"AIC: {fitted_model.aic:.2f}")
    
    return fitted_model


def evaluate_arima_model(model, test, n_steps=24):
    """
    Evaluate ARIMA model on test data
    
    Parameters:
    -----------
    model : ARIMAResults
        Trained ARIMA model
    test : pd.Series
        Test data
    n_steps : int
        Number of steps to forecast
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    actual : np.array
        Actual values
    predictions : np.array
        Predicted values
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Make predictions
    predictions = model.forecast(steps=n_steps)
    
    # Calculate metrics
    mae = mean_absolute_error(test[:n_steps], predictions)
    mse = mean_squared_error(test[:n_steps], predictions)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (avoid division by zero)
    mask = test[:n_steps] != 0
    mape = np.mean(np.abs((test[:n_steps][mask] - predictions[mask]) / test[:n_steps][mask])) * 100 if mask.sum() > 0 else 0
    
    # Calculate R2
    r2 = r2_score(test[:n_steps], predictions)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics, test.values, predictions.values


def save_arima_model(model, model_path='arima_model.pkl'):
    """
    Save trained ARIMA model
    
    Parameters:
    -----------
    model : ARIMAResults or SARIMAXResults
        Trained model
    model_path : str
        Path to save model
    """
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def main():
    """
    Main function to train ARIMA model
    """
    import os
    
    # Configuration
    PROCESSED_UCI = 'processed_uci_hourly.csv'
    PROCESSED_KAGGLE = 'processed_kaggle_hourly.csv'
    
    # Use processed data (ARIMA works with time series directly)
    if os.path.exists(PROCESSED_UCI):
        print("Using UCI dataset for ARIMA...")
        df = pd.read_csv(PROCESSED_UCI, parse_dates=['datetime'], index_col='datetime')
        target_col = 'Global_active_power'
    elif os.path.exists(PROCESSED_KAGGLE):
        print("Using Kaggle dataset for ARIMA...")
        df = pd.read_csv(PROCESSED_KAGGLE, parse_dates=['datetime'], index_col='datetime')
        target_col = df.columns[0]
    else:
        print("No processed data found. Run data_preprocessing.py first.")
        return
    
    print(f"Data shape: {df.shape}")
    print(f"Target column: {target_col}")
    
    # Train model
    model = train_arima_model(df, target_col, order=(1, 1, 1))
    
    # Save model
    save_arima_model(model)
    
    print("\nARIMA model training complete!")


if __name__ == "__main__":
    main()
