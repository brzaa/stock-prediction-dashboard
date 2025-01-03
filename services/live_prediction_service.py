import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import json

# Configure page
st.set_page_config(page_title="Live Stock Prediction Service", page_icon="📈", layout="wide")

def get_gcs_client():
    """Initialize GCS client with credentials"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return storage.Client(credentials=credentials)
    except Exception as e:
        st.error(f"Error initializing GCS client: {str(e)}")
        return None

def fetch_stock_data(bucket):
    """Fetch latest stock data from GCS"""
    try:
        blob = bucket.blob('stock_data/MASB_latest.csv')
        content = blob.download_as_string()
        data = pd.read_csv(pd.io.common.BytesIO(content), parse_dates=['date'])
        data.set_index('date', inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def prepare_tree_based_data(data, train_size=0.8):
    """Prepare data for tree-based models"""
    try:
        # Feature engineering
        data['SMA_5'] = data['close'].rolling(window=5).mean()
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data['close'])
        data['MACD'] = calculate_macd(data['close'])
        data['BB_Upper'], data['BB_Lower'] = calculate_bollinger_bands(data['close'])
        data['Volume_MA5'] = data['volume'].rolling(window=5).mean()
        data['Volume_MA20'] = data['volume'].rolling(window=20).mean()
        data['Price_ROC'] = data['close'].pct_change(periods=5)
        data['Volatility'] = data['close'].rolling(window=10).std()
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        # Split data
        train_size = int(len(data) * train_size)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        feature_cols = [col for col in data.columns if col != 'close']
        X_train = train_data[feature_cols]
        y_train = train_data['close']
        X_test = test_data[feature_cols]
        y_test = test_data['close']
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, test_data.index, feature_cols
    except Exception as e:
        st.error(f"Error preparing tree-based data: {str(e)}")
        return None

def prepare_lstm_data(data, train_size=0.8, sequence_length=20):
    """Prepare data for LSTM model"""
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, dates, features = prepare_tree_based_data(data, train_size)
        
        def create_sequences(X, y, sequence_length):
            Xs, ys = [], []
            for i in range(len(X) - sequence_length):
                Xs.append(X[i:(i + sequence_length)])
                ys.append(y.iloc[i + sequence_length] if isinstance(y, pd.Series) else y[i + sequence_length])
            return np.array(Xs), np.array(ys)
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)
        
        return X_train_seq, X_test_seq, y_train_seq, y_test_seq, dates[sequence_length:], features
    except Exception as e:
        st.error(f"Error preparing LSTM data: {str(e)}")
        return None

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def train_model(model_type, X_train, y_train, X_test, y_test, test_dates, feature_cols, bucket):
    """Train model and save predictions"""
    try:
        with st.spinner(f'Training {model_type} model...'):
            if model_type == 'xgboost':
                params = {
                    "objective": "reg:squarederror",
                    "max_depth": 5,
                    "learning_rate": 0.01,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_type == 'decision_tree':
                params = {
                    "max_depth": 10,
                    "min_samples_split": 20,
                    "min_samples_leaf": 10,
                    "random_state": 42
                }
                model = DecisionTreeRegressor(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    "objective": "regression",
                    "max_depth": 7,
                    "learning_rate": 0.05,
                    "n_estimators": 100,
                    "num_leaves": 31,
                    "random_state": 42
                }
                model = LGBMRegressor(**params)
                
            elif model_type == 'lstm':
                sequence_length = X_train.shape[1]
                feature_dim = X_train.shape[2]
                
                params = {
                    "lstm_units": 100,
                    "dense_units": 50,
                    "dropout_rate": 0.2,
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "batch_size": 32
                }
                
                model = Sequential([
                    LSTM(params["lstm_units"], input_shape=(sequence_length, feature_dim)),
                    Dropout(params["dropout_rate"]),
                    Dense(params["dense_units"], activation='relu'),
                    Dense(1)
                ])
                
                model.compile(optimizer=Adam(learning_rate=params["learning_rate"]),
                            loss='mse')
                
                model.fit(X_train, y_train,
                         epochs=params["epochs"],
                         batch_size=params["batch_size"],
                         validation_split=0.1,
                         verbose=0)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Train model
            if model_type != 'lstm':
                model.fit(X_train, y_train)

            # Generate predictions
            predictions = model.predict(X_test)
            if model_type == 'lstm':
                predictions = predictions.flatten()

            # Calculate metrics
            metrics = {
                'mse': float(mean_squared_error(y_test, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                'r2': float(r2_score(y_test, predictions)),
                'mae': float(mean_absolute_error(y_test, predictions)),
                'mape': float(mean_absolute_percentage_error(y_test, predictions))
            }

            # Prepare results
            results = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'predictions': predictions.tolist(),
                'actual_values': y_test.tolist() if isinstance(y_test, np.ndarray) else y_test.values.tolist(),
                'dates': [str(date) for date in test_dates],
                'metrics': metrics,
                'parameters': params,
                'feature_cols': feature_cols
            }

            # Save predictions to GCS
            blob = bucket.blob(f'live_predictions/{model_type}/latest.json')
            blob.upload_from_string(
                json.dumps(results, indent=2),
                content_type='application/json'
            )

            # Display metrics
            st.subheader(f"{model_type.upper()} Model Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{metrics['mse']:.2f}")
            col2.metric("RMSE", f"{metrics['rmse']:.2f}")
            col3.metric("R²", f"{metrics['r2']:.2f}")
            col4.metric("MAE", f"{metrics['mae']:.2f}")

            return True

    except Exception as e:
        st.error(f"Error training {model_type} model: {str(e)}")
        return False

def train_all_models():
    """Train all models and save predictions"""
    try:
        st.title("🟢 Live Stock Prediction Service")
        
        # Initialize GCS client
        client = get_gcs_client()
        if not client:
            return
            
        bucket = client.bucket("mlops-brza")
        
        # Fetch data
        data = fetch_stock_data(bucket)
        if data is None:
            return

        # Train tree-based models
        X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_tree_based_data(data)
        if X_train is None:
            return
            
        for model_type in ['xgboost', 'decision_tree', 'lightgbm']:
            train_model(model_type, X_train, y_train, X_test, y_test, test_dates, feature_cols, bucket)

        # Train LSTM model
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, test_dates_lstm, feature_cols_lstm = prepare_lstm_data(data)
        if X_train_lstm is not None:
            train_model('lstm', X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, test_dates_lstm, feature_cols_lstm, bucket)

        st.success("✅ All models trained and predictions updated successfully!")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if st.button("🚀 Run Prediction Pipeline"):
        train_all_models()
