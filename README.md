

# ⚡ Energy Consumption Forecasting System

A complete machine learning project for forecasting energy consumption using LSTM deep learning and ARIMA statistical models, with an interactive Streamlit dashboard.

## 📁 Project Structure

```
Energy-Electricity-consumption-forecast/
├── household_power_consumption.txt  # UCI dataset
├── PJME_hourly.csv                 # Kaggle PJM dataset (main)
├── AEP_hourly.csv                  # Kaggle AEP dataset
├── COMED_hourly.csv                # Kaggle COMED dataset
├── DAYTON_hourly.csv               # Kaggle DAYTON dataset
├── DEOK_hourly.csv                 # Kaggle DEOK dataset
├── DOM_hourly.csv                  # Kaggle DOM dataset
├── DUQ_hourly.csv                  # Kaggle DUQ dataset
├── EKPC_hourly.csv                 # Kaggle EKPC dataset
├── FE_hourly.csv                   # Kaggle FE dataset
├── NI_hourly.csv                   # Kaggle NI dataset
├── PJMW_hourly.csv                 # Kaggle PJMW dataset
├── PJM_Load_hourly.csv             # Kaggle PJM Load dataset
├── pjm_hourly_est.csv              # Kaggle PJM estimated data
├── est_hourly.paruqet              # Parquet format data
├── data_preprocessing.py           # Data loading and preprocessing
├── feature_engineering.py          # Feature engineering (temporal, lag, rolling features)
├── lstm_model.py                   # LSTM model training
├── arima_model.py                  # ARIMA/SARIMAX model training
├── model_evaluation.py             # Evaluation metrics (MAE, MSE, RMSE, MAPE, R²)
├── eda_visualization.py            # EDA and visualization script
├── app.py                          # Streamlit UI dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── visualizations/                # Generated HTML plots (empty initially)
```

## 🛠️ Tech Stack

- **Python 3.8+**
- **TensorFlow/Keras** - LSTM deep learning model
- **Statsmodels** - ARIMA/SARIMAX statistical models
- **Streamlit** - Interactive web UI
- **Plotly** - Interactive visualizations
- **Pandas/NumPy** - Data manipulation
- **Scikit-learn** - Metrics and preprocessing

## 📋 Prerequisites

1. Python 3.8 or higher installed
2. Datasets downloaded:
   - UCI Household Power Consumption Dataset (included)
   - Kaggle PJM Hourly Energy Consumption (included)
3. Required packages installed

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Preprocess Data

```bash
python data_preprocessing.py
```

This will:
- Load UCI and/or Kaggle datasets from current directory
- Handle missing values
- Resample UCI data from 1-minute to hourly
- Save processed data to current directory

### Step 3: Exploratory Data Analysis

```bash
python eda_visualization.py
```

This will generate interactive HTML visualizations in the `visualizations/` directory:
- Time series plots
- Seasonal decomposition
- Hourly/daily/monthly patterns
- Heatmaps
- Correlation matrices
- Distribution plots

### Step 4: Feature Engineering (Optional)

```bash
python feature_engineering.py
```

This creates additional features:
- Temporal features (hour, day, month, cyclical encoding)
- Lag features (1h, 24h, 168h)
- Rolling statistics (mean, std, min, max)
- Difference features
- Interaction features+

### Step 5: Train Models

**Train LSTM Model:**
```bash
python lstm_model.py
```

**Train ARIMA Model:**
```bash
python arima_model.py
```

This will:
- Train the models on processed data
- Evaluate performance (MAE, RMSE, MAPE, R²)
- Save trained models to current directory

### Step 6: Evaluate Models (Optional)

```bash
python model_evaluation.py
```

This will:
- Load trained models
- Calculate comprehensive metrics (MAE, MSE, RMSE, MAPE, R²)
- Generate prediction plots and residual analysis
- Compare multiple models side-by-side
- Save metrics to files

### Step 7: Launch Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Streamlit Dashboard Features

### 🏠 Dashboard Page
- Key metrics (total, average, peak consumption)
- Interactive time series plot with date range selector
- Hourly and daily consumption patterns
- Data exploration

### 🔮 Forecast Page
- Select forecast horizon (24h, 48h, 72h, 168h)
- Choose start date
- Generate predictions
- View forecast with confidence intervals
- Download forecasts as CSV

### 🎯 Model Performance Page
- Model accuracy metrics comparison
- Actual vs Predicted plots
- Residual analysis

### 📊 Data Explorer Page
- Dataset information and statistics
- Raw data preview
- Download full dataset

## 📈 Model Details

### LSTM Model (Deep Learning)
- Architecture: LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1)
- Input: 24-hour lookback window
- Training: Adam optimizer, MSE loss
- Best for: Capturing complex temporal patterns

### ARIMA Model (Statistical)
- Default: ARIMA(1,1,1)
- Optional: SARIMAX for seasonal patterns
- Optional: Grid search for optimal parameters
- Best for: Baseline comparison, interpretable results

## 📁 Datasets

### UCI Household Power Consumption Dataset
- **Source**: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
- **Size**: 2M+ measurements (2006-2010)
- **Frequency**: 1-minute → Resampled to hourly
- **Features**: Global active power, voltage, sub-metering data

### Kaggle PJM Hourly Energy Consumption
- **Source**: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- **Size**: Multiple regions (PJM, AEP, COMED, etc.)
- **Frequency**: Already hourly
- **Features**: Hourly consumption in MW

## 🎯 Workflow

```
1. Data Preprocessing
   ↓
2. EDA & Visualizations
   ↓
3. Feature Engineering (Optional)
   ↓
4. Model Training (LSTM + ARIMA)
   ↓
5. Model Evaluation (Optional)
   ↓
6. Streamlit Dashboard
```

## 📊 Evaluation Metrics

- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values
- **MSE** (Mean Squared Error): Average of squared differences between predictions and actual values
- **RMSE** (Root Mean Squared Error): Square root of average squared differences
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
- **R²** (R-squared): Proportion of variance explained by the model

## 🔧 Configuration

Edit the configuration variables in each script to customize:
- Dataset paths (scripts now reference `Datasets/` directory)
- Model hyperparameters (epochs, batch size, LSTM units)
- ARIMA parameters (p, d, q)
- Feature engineering options (lags, windows)

**Note**: All scripts have been updated to use the new directory structure:
- Datasets are located in `Datasets/` directory
- Preprocessing scripts are in `DataPreprocessing/` directory
- Model scripts are in `models/` directory
- Evaluation scripts are in `Evaluation_metrics/` directory
- Visualization scripts are in `visualizations/` directory

## 🚧 Future Enhancements

- [ ] Weather data integration (OpenWeatherMap API)
- [ ] XGBoost model implementation
- [ ] Prophet model implementation
- [ ] Ensemble model (combining multiple models)
- [ ] Real-time prediction streaming
- [ ] Anomaly detection alerts
- [ ] Model retraining pipeline

## 📝 Notes

- The LSTM model requires TensorFlow 2.12+
- Training time depends on dataset size and hardware
- GPU recommended for LSTM training on large datasets
- Streamlit dashboard runs on CPU (no GPU needed)

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: `No data found` error
- **Solution**: Run `DataPreprocessing/data_preprocessing.py` first to process datasets from `Datasets/` directory

**Issue**: LSTM training is slow
- **Solution**: Reduce epochs or batch size, or use GPU

**Issue**: Streamlit not opening
- **Solution**: Check if port 8501 is available, or run `streamlit run app.py --server.port 8502`

**Issue**: Scripts can't find datasets
- **Solution**: Ensure all dataset files are in the `Datasets/` directory

## 📄 License

This project is for educational purposes.

## 👤 Author

Energy Consumption Forecasting Project

---

**Happy Forecasting! ⚡**

=======
# Energy-Electricity-consumption-forecast
ML Mini Project

