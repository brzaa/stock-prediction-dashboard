import os
import pandas as pd
import numpy as np
from google.cloud import storage
import logging
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import json

# Configure logging
logging.basicConfig(
    filename='live_predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockPredictor:
    def __init__(self, bucket_name="mlops-brza"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        self.models = {
            'xgboost': self.train_xgboost,
            'decision_tree': self.train_decision_tree,
            'lightgbm': self.train_lightgbm,
            'lstm': self.train_lstm
        }
    
    def fetch_stock_data(self):
        """Fetch stock data from local file in GCS"""
        try:
            logger.info("Fetching stock data from GCS.")
            blob = self.bucket.blob('stock_data/MASB_latest.csv')
            local_file = '/tmp/MASB_latest.csv'
            blob.download_to_filename(local_file)
            data = pd.read_csv(local_file, parse_dates=['date'], index_col='date')
            logger.info("Stock data fetched successfully.")
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def prepare_data(self, data: pd.DataFrame):
        """Prepare data for training"""
        logger.info("Preparing data for training.")
        feature_cols = [col for col in data.columns if col not in ['close']]
        
        # Split data into training and testing
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        X_train = train_data[feature_cols]
        y_train = train_data['close']
        X_test = test_data[feature_cols]
        y_test = test_data['close']
        
        # Scale data for LSTM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Data preparation completed.")
        return (X_train_scaled, X_test_scaled, y_train, y_test, test_data.index, feature_cols)

    def train_xgboost(self, X_train, y_train):
        """Train an XGBoost model"""
        params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 500
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("XGBoost model trained successfully.")
        return model, params

    def train_decision_tree(self, X_train, y_train):
        """Train a Decision Tree model"""
        params = {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10
        }
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("Decision Tree model trained successfully.")
        return model, params

    def train_lightgbm(self, X_train, y_train):
        """Train a LightGBM model"""
        params = {
            "objective": "regression",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "num_leaves": 31
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("LightGBM model trained successfully.")
        return model, params

    def train_lstm(self, X_train, y_train):
        """Train an LSTM model"""
        time_steps = 5
        
        def create_sequences(X, y):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train.values)
        
        model = Sequential([
            Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
        
        params = {
            "lstm_units_1": 64,
            "lstm_units_2": 32,
            "dropout_rate": 0.2,
            "time_steps": time_steps
        }
        
        logger.info("LSTM model trained successfully.")
        return model, params

    def run_predictions(self):
        """Run predictions for all models."""
        logger.info("Starting predictions for all models.")
        try:
            data = self.fetch_stock_data()
            X_train, X_test, y_train, y_test, test_dates, feature_cols = self.prepare_data(data)

            for model_type, train_func in self.models.items():
                try:
                    logger.info(f"Training and predicting with {model_type}...")
                    model, params = train_func(X_train, y_train)
                    predictions = model.predict(X_test) if model_type != 'lstm' else model.predict(X_test).flatten()

                    metrics = {
                        'mse': float(mean_squared_error(y_test, predictions)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                        'r2': float(r2_score(y_test, predictions)),
                        'mae': float(mean_absolute_error(y_test, predictions))
                    }

                    results = {
                        'timestamp': datetime.now().isoformat(),
                        'model_type': model_type,
                        'predictions': predictions.tolist(),
                        'actual_values': y_test.tolist(),
                        'dates': [d.strftime('%Y-%m-%d') for d in test_dates],
                        'metrics': metrics,
                        'parameters': params,
                        'feature_names': feature_cols
                    }

                    logger.info(f"Saving predictions for {model_type} to GCS...")
                    blob = self.bucket.blob(f'live_predictions/{model_type}/latest.json')
                    blob.upload_from_string(json.dumps(results), content_type='application/json')
                    logger.info(f"Predictions saved for {model_type}.")
                except Exception as e:
                    logger.error(f"Error in {model_type} predictions: {str(e)}")
        except Exception as e:
            logger.error(f"Error in run_predictions: {str(e)}")


if __name__ == "__main__":
    predictor = StockPredictor()
    predictor.run_predictions()
