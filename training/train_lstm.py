import os
import mlflow
import mlflow.keras
import pandas as pd
import talib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from google.cloud import storage
import logging
from typing import Tuple, Any
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_model_results(run_id: str, model_type: str, results: dict, bucket_name: str = "mlops-brza"):
    """Save model results organized by model type and date"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Organize results by model type
        output_path = f"model_outputs/{model_type}/{run_id}/results.json"
        blob = bucket.blob(output_path)
        blob.upload_from_string(
            json.dumps(results, indent=2),
            content_type='application/json'
        )
        
        # Also save a summary for easy lookup
        summary = {
            'run_id': run_id,
            'model_type': model_type,
            'timestamp': results['timestamp'],
            'metrics': results['metrics']
        }
        summary_blob = bucket.blob(f'model_outputs/summary/{run_id}.json')
        summary_blob.upload_from_string(json.dumps(summary, indent=2))
        
        logger.info(f"Results saved to: gs://{bucket_name}/{output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

def create_lstm_model(input_shape: Tuple[int, int], params: dict) -> Sequential:
    """Create LSTM model with given parameters"""
    model = Sequential([
        LSTM(params['lstm1_units'], return_sequences=True, input_shape=input_shape),
        Dropout(params['dropout1']),
        LSTM(params['lstm2_units']),
        Dropout(params['dropout2']),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

def train_and_log_model_colab(
    project_id: str,
    bucket_name: str = "mlops-brza",
    split_date: str = '2022-01-01',
    model_type: str = "lstm"
) -> Tuple[str, Any, Any]:
    """Train LSTM model and log with MLflow in Colab"""

    with mlflow.start_run() as run:
        logger.info(f"Starting training for model type: {model_type}")
        logger.info("Fetching data from GCS...")

        # Initialize GCS client and bucket
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(bucket_name)

        # Fetch data
        blob = bucket.blob('stock_data/MASB.csv')
        data = pd.read_csv(blob.open(mode='rb'), parse_dates=['date'], index_col='date')

        # Data preprocessing
        data.columns = data.columns.str.lower()
        data['close'] = pd.to_numeric(data['close'], errors='coerce')

        # Feature engineering
        data['moving_avg_5'] = data['close'].rolling(window=5).mean()
        data['rsi'] = talib.RSI(data['close'].values, timeperiod=14)
        data['macd'], data['signal'], data['hist'] = talib.MACD(data['close'].values)
        data = data.ffill().bfill().dropna()

        # Split data
        split_date = pd.Timestamp(split_date)
        train_data = data.loc[data.index < split_date]
        test_data = data.loc[data.index >= split_date]

        # Scale data
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train_data)
        scaled_test = scaler.transform(test_data)

        # Prepare features and target
        X_train = scaled_train[:, :-1]
        y_train = scaled_train[:, data.columns.get_loc('close')]
        X_test = scaled_test[:, :-1]
        y_test = scaled_test[:, data.columns.get_loc('close')]

        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Log dataset info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("split_date", split_date)

        # Model parameters
        params = {
            'lstm1_units': 64,
            'lstm2_units': 32,
            'dropout1': 0.2,
            'dropout2': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 50
        }

        # Log parameters
        mlflow.log_params(params)

        # Create and train model
        model = create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            params=params
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # Log metrics
        mlflow.log_metrics({
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })

        # Save comprehensive results
        results = {
            'run_id': run.info.run_id,
            'model_type': model_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mae': float(mae)
            },
            'parameters': params,
            'predictions': y_pred.flatten().tolist(),
            'actual_values': y_test.tolist(),
            'dates': test_data.index.strftime('%Y-%m-%d').tolist()
        }

        save_model_results(run.info.run_id, model_type, results, bucket_name)

        # Save model locally first
        model_path = "/content/models"
        os.makedirs(model_path, exist_ok=True)

        # Save model to GCS
        model.save(f"{model_path}/{model_type}_model.keras")
        model_blob = bucket.blob(f'models/{model_type}/{run.info.run_id}/model.keras')
        model_blob.upload_from_filename(f"{model_path}/{model_type}_model.keras")

        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
        print(f"Run ID to use in dashboard: {run.info.run_id}")

        return run.info.run_id, scaler, model, X_test, y_test

if __name__ == "__main__":
    project_id = "mlops-thesis"
    
    # Train multiple model types or variations
    model_types = [
        "lstm_default",
        "lstm_optimized",
        "lstm_advanced"  # Add more model types as needed
    ]
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        run_id, _, _, _, _ = train_and_log_model_colab(
            project_id=project_id,
            model_type=model_type
        )
        print(f"Completed {model_type} training. Run ID: {run_id}\n")
