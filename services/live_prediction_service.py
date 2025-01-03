import os
# TensorFlow and CUDA configuration (must be first)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN

# Suppress tensorflow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import tensorflow as tf
from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Live Stock Prediction Service",
    page_icon="📈",
    layout="wide"
)

def get_gcs_client():
    """Initialize GCS client with credentials"""
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return storage.Client(credentials=credentials)

def prepare_data(data):
    """Prepare and engineer features"""
    df = data.copy()
    
    # Technical indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['Price_ROC'] = df['close'].pct_change(periods=5)
    df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def train_and_save_model(model_name, model, X_train, X_test, y_train, y_test, 
                        test_dates, feature_cols, bucket, progress_bar):
    """Train model and save predictions"""
    # Update progress
    progress_bar.progress((list(MODELS.keys()).index(model_name) + 1) / len(MODELS))
    st.text(f"Training {model_name} model...")
    
    # Train model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
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
        'model_type': model_name,
        'predictions': predictions.tolist(),
        'actual_values': y_test.tolist(),
        'dates': [str(date) for date in test_dates],
        'metrics': metrics,
        'feature_cols': feature_cols
    }
    
    # Save to GCS
    blob = bucket.blob(f'live_predictions/{model_name}/latest.json')
    blob.upload_from_string(json.dumps(results, indent=2))
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{model_name.upper()} - MSE", f"{metrics['mse']:.2f}")
    with col2:
        st.metric(f"{model_name.upper()} - RMSE", f"{metrics['rmse']:.2f}")
    with col3:
        st.metric(f"{model_name.upper()} - R²", f"{metrics['r2']:.2f}")
    with col4:
        st.metric(f"{model_name.upper()} - MAE", f"{metrics['mae']:.2f}")

# Define models
MODELS = {
    'xgboost': xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=5,
        learning_rate=0.01,
        n_estimators=100,
        random_state=42
    ),
    'decision_tree': DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=20,
        random_state=42
    ),
    'lightgbm': LGBMRegressor(
        objective="regression",
        max_depth=7,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42
    )
}

def main():
    st.title("🟢 Live Stock Prediction Service")
    st.markdown("""
    This service updates predictions for multiple models:
    - XGBoost
    - Decision Tree
    - LightGBM
    """)
    
    if st.button("🚀 Run Prediction Pipeline"):
        try:
            # Initialize progress
            progress_bar = st.progress(0)
            status = st.empty()
            
            # Setup GCS client
            status.text("Connecting to Google Cloud Storage...")
            client = get_gcs_client()
            bucket = client.bucket("mlops-brza")
            
            # Load data
            status.text("Loading latest stock data...")
            blob = bucket.blob('stock_data/MASB_latest.csv')
            content = blob.download_as_string()
            data = pd.read_csv(pd.io.common.BytesIO(content), parse_dates=['date'])
            data.set_index('date', inplace=True)
            
            # Prepare data
            status.text("Preparing features...")
            processed_data = prepare_data(data)
            
            # Split data
            train_size = int(len(processed_data) * 0.8)
            feature_cols = ['SMA_5', 'SMA_20', 'Price_ROC', 'Volume_MA5', 'Volume_MA20', 'volume']
            
            X = processed_data[feature_cols]
            y = processed_data['close']
            
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train = X_scaled[:train_size]
            X_test = X_scaled[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            test_dates = processed_data.index[train_size:]
            
            # Train and save each model
            for model_name, model in MODELS.items():
                train_and_save_model(
                    model_name, model, 
                    X_train, X_test, 
                    y_train, y_test, 
                    test_dates, feature_cols,
                    bucket, progress_bar
                )
            
            # Complete
            progress_bar.progress(1.0)
            status.success("✅ All models trained and predictions updated successfully!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            raise e

if __name__ == "__main__":
    main()
