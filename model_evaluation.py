"""
Model Evaluation Script
Calculate and display evaluation metrics for trained models
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    metrics : dict
        Dictionary of metrics (MAE, MSE, RMSE, MAPE, R2)
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    # Calculate R2
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics


def evaluate_lstm_model(model, scaler, df, target_col, seq_length=24):
    """
    Evaluate LSTM model
    
    Parameters:
    -----------
    model : Sequential
        Trained LSTM model
    scaler : MinMaxScaler
        Fitted scaler
    df : pd.DataFrame
        Test data
    target_col : str
        Target column name
    seq_length : int
        Sequence length for LSTM
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    actual : np.array
        Actual values
    predictions : np.array
        Predicted values
    """
    # Prepare data
    data = df[target_col].values.reshape(-1, 1)
    data_scaled = scaler.transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:(i + seq_length)])
        y.append(data_scaled[i + seq_length])
    X, y = np.array(X), np.array(y)
    
    # Make predictions
    y_pred_scaled = model.predict(X)
    
    # Inverse transform
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y)
    
    # Calculate metrics
    metrics = calculate_metrics(y_actual, y_pred)
    
    print("\nLSTM Model Performance:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics, y_actual, y_pred


def evaluate_arima_model(model, df, target_col, n_steps=24):
    """
    Evaluate ARIMA model
    
    Parameters:
    -----------
    model : ARIMAResults
        Trained ARIMA model
    df : pd.DataFrame
        Test data
    target_col : str
        Target column name
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
    # Get actual values
    actual = df[target_col].values[:n_steps]
    
    # Make predictions
    predictions = model.forecast(steps=n_steps)
    
    # Calculate metrics
    metrics = calculate_metrics(actual, predictions)
    
    print("\nARIMA Model Performance:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics, actual, predictions.values


def plot_predictions(actual, predictions, title="Actual vs Predicted", show_plot=False):
    """
    Plot actual vs predicted values
    
    Parameters:
    -----------
    actual : np.array
        Actual values
    predictions : np.array
        Predicted values
    title : str
        Plot title
    show_plot : bool
        Whether to display the plot
    """
    if not show_plot:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_residuals(actual, predictions, show_plot=False):
    """
    Plot residuals
    
    Parameters:
    -----------
    actual : np.array
        Actual values
    predictions : np.array
        Predicted values
    show_plot : bool
        Whether to display the plot
    """
    if not show_plot:
        return
    
    residuals = actual - predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals plot
    axes[0].plot(residuals)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title('Residuals Over Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Residual')
    axes[0].grid(True)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_title('Residuals Distribution')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def save_metrics_to_file(metrics_dict, output_path='evaluation_metrics.txt'):
    """
    Save evaluation metrics to a text file
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing metrics for different models
    output_path : str
        Path to save the metrics file
    """
    with open(output_path, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, metrics in metrics_dict.items():
            f.write(f"{model_name}:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
    
    print(f"Metrics saved to: {output_path}")


def main():
    """
    Main function to run model evaluation
    """
    print("Model Evaluation Script")
    print("=" * 60)
    
    # Configuration
    LSTM_MODEL_PATH = 'lstm_model.keras'
    LSTM_SCALER_PATH = 'scaler.pkl'
    ARIMA_MODEL_PATH = 'arima_model.pkl'
    DATA_PATH = 'processed_uci_hourly.csv'
    
    metrics_dict = {}
    
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'], index_col='datetime')
    target_col = 'Global_active_power' if 'Global_active_power' in df.columns else df.columns[0]
    print(f"Data loaded: {df.shape}")
    print(f"Target column: {target_col}")
    
    # Evaluate LSTM model
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
        print("\nEvaluating LSTM model...")
        try:
            model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            scaler = joblib.load(LSTM_SCALER_PATH)
            metrics, actual, predictions = evaluate_lstm_model(model, scaler, df, target_col)
            metrics_dict['LSTM'] = metrics
            # Skip plotting in headless environment
        except Exception as e:
            print(f"Error evaluating LSTM model: {e}")
    else:
        print("LSTM model files not found. Train the model first.")
    
    # Evaluate ARIMA model
    if os.path.exists(ARIMA_MODEL_PATH):
        print("\nEvaluating ARIMA model...")
        try:
            model = joblib.load(ARIMA_MODEL_PATH)
            metrics, actual, predictions = evaluate_arima_model(model, df, target_col)
            metrics_dict['ARIMA'] = metrics
            # Skip plotting in headless environment
        except Exception as e:
            print(f"Error evaluating ARIMA model: {e}")
    else:
        print("ARIMA model file not found. Train the model first.")
    
    # Save metrics
    if metrics_dict:
        save_metrics_to_file(metrics_dict)
        print("\nEvaluation complete!")
    else:
        print("\nNo models evaluated. Train the models first.")


if __name__ == "__main__":
    main()
