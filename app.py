"""
Streamlit UI for Energy Consumption Forecasting
Interactive dashboard for predictions and visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import tensorflow as tf
import joblib
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to decrease font size
st.markdown("""
<style>
    body {
        font-size: 12px !important;
    }
    h1 {
        font-size: 24px !important;
    }
    h2 {
        font-size: 20px !important;
    }
    h3 {
        font-size: 18px !important;
    }
    p {
        font-size: 12px !important;
    }
    .stMarkdown {
        font-size: 12px !important;
    }
    .metric-label {
        font-size: 11px !important;
    }
    .metric-value {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data(filepath):
    """Load processed data"""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
        return df
    return None


@st.cache_resource
def load_lstm_model():
    """Load trained LSTM model"""
    try:
        model = tf.keras.models.load_model('lstm_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None


@st.cache_resource
def load_arima_model():
    """Load trained ARIMA model"""
    try:
        model = joblib.load('arima_model.pkl')
        return model
    except:
        return None


def plot_time_series(df, column, title="Energy Consumption Over Time"):
    """Plot time series chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode='lines',
        name='Consumption',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Energy (kW)',
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_prediction(actual, predicted, dates):
    """Plot actual vs predicted values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Consumption',
        xaxis_title='Date',
        yaxis_title='Energy (kW)',
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_forecast(forecast_values, forecast_dates):
    """Plot forecast with confidence intervals"""
    fig = go.Figure()
    
    # Main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    
    # Add confidence interval (simulated)
    lower_bound = forecast_values * 0.9
    upper_bound = forecast_values * 1.1
    
    fig.add_trace(go.Scatter(
        x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Confidence Interval (±10%)'
    ))
    
    fig.update_layout(
        title='Energy Consumption Forecast',
        xaxis_title='Date',
        yaxis_title='Energy (kW)',
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_hourly_pattern(df, column):
    """Plot average consumption by hour"""
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    hourly_mean = df_copy.groupby('hour')[column].mean()
    
    fig = go.Figure(data=go.Bar(
        x=hourly_mean.index,
        y=hourly_mean.values,
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title='Average Consumption by Hour of Day',
        xaxis_title='Hour (0-23)',
        yaxis_title='Average Consumption (kW)',
        template='plotly_dark',
        height=350
    )
    
    return fig


def plot_daily_pattern(df, column):
    """Plot average consumption by day of week"""
    df_copy = df.copy()
    df_copy['day_name'] = df_copy.index.day_name()
    daily_mean = df_copy.groupby('day_name')[column].mean()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_mean = daily_mean.reindex(day_order)
    
    fig = go.Figure(data=go.Bar(
        x=daily_mean.index,
        y=daily_mean.values,
        marker_color='#d62728'
    ))
    
    fig.update_layout(
        title='Average Consumption by Day of Week',
        xaxis_title='Day',
        yaxis_title='Average Consumption (kW)',
        template='plotly_dark',
        height=350
    )
    
    return fig


def main():
    """Main Streamlit app"""
    
    # Sidebar
    st.sidebar.title("⚡ Energy Forecaster")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Dashboard", "🔮 Forecast", "🎯 Model Performance", "📊 Data Explorer"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose Model",
        ["LSTM (Deep Learning)", "ARIMA (Statistical)", "Both (Comparison)"]
    )
    
    # Dashboard Page
    if page == "🏠 Dashboard":
        st.title("⚡ Energy Consumption Dashboard")
        
        # Load data
        data_path = 'processed_uci_hourly.csv' if os.path.exists('processed_uci_hourly.csv') else 'processed_kaggle_hourly.csv'
        df = load_data(data_path)
        
        if df is None:
            st.error("No data found. Please run data_preprocessing.py first.")
            return
        
        # Get target column
        if 'Global_active_power' in df.columns:
            target_col = 'Global_active_power'
        else:
            target_col = df.columns[0]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_consumption = df[target_col].sum()
            st.metric("Total Consumption", f"{total_consumption:.2f} kW")
        
        with col2:
            avg_consumption = df[target_col].mean()
            st.metric("Average Consumption", f"{avg_consumption:.2f} kW/h")
        
        with col3:
            peak_consumption = df[target_col].max()
            st.metric("Peak Consumption", f"{peak_consumption:.2f} kW")
        
        with col4:
            data_points = len(df)
            st.metric("Data Points", f"{data_points:,}")
        
        st.markdown("---")
        
        # Time series plot
        st.subheader("📈 Consumption Over Time")
        
        # Date range selector
        date_range = st.slider(
            "Select Date Range",
            min_value=df.index.min().date(),
            max_value=df.index.max().date(),
            value=(df.index.min().date(), df.index.max().date())
        )
        
        mask = (df.index.date >= date_range[0]) & (df.index.date <= date_range[1])
        df_filtered = df[mask]
        
        fig = plot_time_series(df_filtered, target_col)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Pattern analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏰ Hourly Pattern")
            fig_hourly = plot_hourly_pattern(df_filtered, target_col)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            st.subheader("📅 Daily Pattern")
            fig_daily = plot_daily_pattern(df_filtered, target_col)
            st.plotly_chart(fig_daily, use_container_width=True)
    
    # Forecast Page
    elif page == "🔮 Forecast":
        st.title("🔮 Energy Consumption Forecast")
        
        # Load data and models
        data_path = 'processed_uci_hourly.csv' if os.path.exists('processed_uci_hourly.csv') else 'processed_kaggle_hourly.csv'
        df = load_data(data_path)
        
        if df is None:
            st.error("No data found. Please run data_preprocessing.py first.")
            return
        
        lstm_model, scaler = load_lstm_model()
        arima_model = load_arima_model()
        
        # Forecast parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_hours = st.selectbox(
                "Forecast Horizon",
                [24, 48, 72, 168],  # 1 day, 2 days, 3 days, 1 week
                index=0
            )
        
        with col2:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date()
            )
        
        with col3:
            st.write("")
            st.write("")
            generate_btn = st.button("🚀 Generate Forecast", type="primary")
        
        if generate_btn:
            st.markdown("---")
            
            # Get target column
            if 'Global_active_power' in df.columns:
                target_col = 'Global_active_power'
            else:
                target_col = df.columns[0]
            
            # Generate forecast dates
            forecast_dates = pd.date_range(start=start_date, periods=forecast_hours, freq='H')
            
            # Simulate forecast (replace with actual model prediction)
            # For demo, we'll use the average of recent data with some variation
            recent_data = df[target_col].tail(24).values
            base_value = recent_data.mean()
            
            # Add daily pattern
            forecast_values = []
            for i, date in enumerate(forecast_dates):
                hour = date.hour
                # Simulate daily pattern (higher in evening, lower at night)
                hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 18) / 24)
                # Add some randomness
                noise = np.random.normal(0, 0.1 * base_value)
                value = base_value * hour_factor + noise
                forecast_values.append(max(value, 0))  # Ensure non-negative
            
            forecast_values = np.array(forecast_values)
            
            # Display forecast
            st.subheader("📊 Forecast Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_forecast = forecast_values.sum()
                st.metric("Total Forecast", f"{total_forecast:.2f} kW")
            
            with col2:
                avg_forecast = forecast_values.mean()
                st.metric("Average", f"{avg_forecast:.2f} kW/h")
            
            with col3:
                peak_forecast = forecast_values.max()
                peak_hour = forecast_dates[np.argmax(forecast_values)].hour
                st.metric("Peak", f"{peak_forecast:.2f} kW at {peak_hour}:00")
            
            # Plot forecast
            fig = plot_forecast(forecast_values, forecast_dates)
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.subheader("📋 Detailed Forecast")
            
            forecast_df = pd.DataFrame({
                'DateTime': forecast_dates,
                'Predicted Consumption (kW)': forecast_values,
                'Low Estimate (kW)': forecast_values * 0.9,
                'High Estimate (kW)': forecast_values * 1.1
            })
            
            st.dataframe(forecast_df, use_container_width=True, height=400)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"forecast_{start_date}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.info("💡 Tip: Train the LSTM model to get accurate predictions instead of this demo forecast.")
    
    # Model Performance Page
    elif page == "🎯 Model Performance":
        st.title("🎯 Model Performance")
        
        # Try to load metrics from file
        metrics_file = 'evaluation_metrics.txt'
        if os.path.exists(metrics_file):
            st.success("📊 Model evaluation results loaded!")
            
            # Parse metrics file
            with open(metrics_file, 'r') as f:
                content = f.read()
            
            # Parse metrics into dictionary
            metrics_dict = {}
            current_model = None
            for line in content.split('\n'):
                if line.strip() and ':' in line:
                    if 'LSTM' in line or 'ARIMA' in line:
                        current_model = line.replace(':', '').strip()
                        metrics_dict[current_model] = {}
                    elif current_model:
                        key, value = line.split(':', 1)
                        metrics_dict[current_model][key.strip()] = float(value.strip())
            
            # Create comparison table
            st.markdown("---")
            st.subheader("Model Accuracy Metrics Comparison")
            
            if metrics_dict:
                comparison_data = {
                    'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE', 'R²']
                }
                
                for model_name in metrics_dict.keys():
                    comparison_data[model_name] = [
                        metrics_dict[model_name].get('MAE', '-'),
                        metrics_dict[model_name].get('MSE', '-'),
                        metrics_dict[model_name].get('RMSE', '-'),
                        metrics_dict[model_name].get('MAPE', '-'),
                        metrics_dict[model_name].get('R2', '-')
                    ]
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Highlight best performing model
                st.markdown("---")
                st.subheader("💡 Insights")
                if 'LSTM' in metrics_dict and 'ARIMA' in metrics_dict:
                    lstm_mae = metrics_dict['LSTM'].get('MAE', float('inf'))
                    arima_mae = metrics_dict['ARIMA'].get('MAE', float('inf'))
                    
                    if lstm_mae < arima_mae:
                        st.success("🏆 LSTM model performs better (lower MAE indicates better accuracy)")
                    else:
                        st.success("🏆 ARIMA model performs better (lower MAE indicates better accuracy)")
            else:
                st.warning("Could not parse metrics from file.")
        else:
            # Check if models exist
            lstm_exists = os.path.exists('lstm_model.keras') and os.path.exists('scaler.pkl')
            arima_exists = os.path.exists('arima_model.pkl')
            
            if lstm_exists or arima_exists:
                st.warning("📊 Models trained but not evaluated. Run `python model_evaluation.py` to generate metrics.")
            else:
                st.info("📊 Train the models (lstm_model.py and arima_model.py) to see performance metrics here.")
            
            # Placeholder for model comparison
            st.markdown("---")
            st.subheader("Model Accuracy Metrics")
            
            # Create a placeholder metrics table
            metrics_data = {
                'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE', 'R²'],
                'LSTM': ['-', '-', '-', '-', '-'],
                'ARIMA': ['-', '-', '-', '-', '-']
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Data Explorer Page
    elif page == "📊 Data Explorer":
        st.title("📊 Data Explorer")
        
        # Load data
        data_path = 'processed_uci_hourly.csv' if os.path.exists('processed_uci_hourly.csv') else 'processed_kaggle_hourly.csv'
        df = load_data(data_path)
        
        if df is None:
            st.error("No data found. Please run data_preprocessing.py first.")
            return
        
        # Data info
        st.subheader("Dataset Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Records", f"{len(df):,}")
        
        with col2:
            st.metric("Date Range", f"{df.index.min().date()} to {df.index.max().date()}")
        
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum()}")
        
        st.markdown("---")
        
        # Raw data preview
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True, height=400)
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Download full dataset
        st.markdown("---")
        csv = df.to_csv()
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv,
            file_name="energy_consumption_data.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
