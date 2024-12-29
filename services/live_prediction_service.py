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
        self.scaler = StandardScaler()
        
        # Dictionary to store model-specific data preparation methods
        self.data_prep_methods = {
            'xgboost': self._prepare_tree_based_data,
            'decision_tree': self._prepare_tree_based_data,
            'lightgbm': self._prepare_tree_based_data,
            'lstm': self._prepare_lstm_data
        }
        
        # Dictionary to store model training methods
        self.models = {
            'xgboost': self.train_xgboost,
            'decision_tree': self.train_decision_tree,
            'lightgbm': self.train_lightgbm,
            'lstm': self.train_lstm
        }

    def fetch_stock_data(self):
        """Fetch stock data from GCS"""
        try:
            logger.info("Fetching stock data from GCS")
            blob = self.bucket.blob('stock_data/MASB_latest.csv')
            local_file = '/tmp/MASB_latest.csv'
            blob.download_to_filename(local_file)
            data = pd.read_csv(local_file, parse_dates=['date'], index_col='date')
            logger.info(f"Stock data fetched successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def _prepare_tree_based_data(self, data, train_size=0.8):
        """Prepare data for tree-based models"""
        feature_cols = [col for col in data.columns if col != 'close']
        
        train_data = data[:int(len(data) * train_size)]
        test_data = data[int(len(data) * train_size):]
        
        X_train = train_data[feature_cols]
        y_train = train_data['close']
        X_test = test_data[feature_cols]
        y_test = test_data['close']
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, test_data.index, feature_cols

    def _prepare_lstm_data(self, data, train_size=0.8, time_steps=5):
        """Prepare data specifically for LSTM"""
        X_train_scaled, X_test_scaled, y_train, y_test, test_dates, feature_cols = self._prepare_tree_based_data(data, train_size)
        
        # Create sequences for LSTM
        def create_sequences(X, y):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y.iloc[i + time_steps] if isinstance(y, pd.Series) else y[i + time_steps])
            return np.array(Xs), np.array(ys)
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)
        
        return X_train_seq, X_test_seq, y_train_seq, y_test_seq, test_dates[time_steps:], feature_cols

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 500
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("XGBoost model trained successfully")
        return model, params

    def train_decision_tree(self, X_train, y_train):
        """Train Decision Tree model"""
        params = {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10
        }
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("Decision Tree model trained successfully")
        return model, params

    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        params = {
            "objective": "regression",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "num_leaves": 31
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("LightGBM model trained successfully")
        return model, params

    def train_lstm(self, X_train, y_train):
        """Train LSTM model"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        params = {
            "lstm_units_1": 64,
            "lstm_units_2": 32,
            "dropout_rate": 0.2,
            "time_steps": X_train.shape[1]
        }
        
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params["lstm_units_1"], return_sequences=True),
            Dropout(params["dropout_rate"]),
            LSTM(params["lstm_units_2"]),
            Dropout(params["dropout_rate"]),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        logger.info("LSTM model trained successfully")
        return model, params

    def predict_model(self, model, X_test, model_type):
        """Make predictions based on model type"""
        try:
            if model_type == 'lstm':
                predictions = model.predict(X_test).flatten()
            else:
                predictions = model.predict(X_test)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions for {model_type}: {str(e)}")
            raise

    def run_predictions(self, model_types=None):
        """Run predictions for specified models or all models"""
        if model_types is None:
            model_types = list(self.models.keys())
        elif isinstance(model_types, str):
            model_types = [model_types]
            
        logger.info(f"Starting predictions for models: {model_types}")
        
        try:
            data = self.fetch_stock_data()
            
            for model_type in model_types:
                try:
                    logger.info(f"Processing {model_type} model")
                    
                    # Prepare data according to model type
                    prepare_data = self.data_prep_methods[model_type]
                    if model_type == 'lstm':
                        X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_data(data)
                    else:
                        X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_data(data)
                    
                    # Train model and make predictions
                    model, params = self.models[model_type](X_train, y_train)
                    predictions = self.predict_model(model, X_test, model_type)
                    
                    # Calculate metrics
                    metrics = {
                        'mse': float(mean_squared_error(y_test, predictions)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                        'r2': float(r2_score(y_test, predictions)),
                        'mae': float(mean_absolute_error(y_test, predictions))
                    }
                    
                    # Prepare results
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
                    
                    # Save predictions
                    blob = self.bucket.blob(f'live_predictions/{model_type}/latest.json')
                    blob.upload_from_string(
                        json.dumps(results),
                        content_type='application/json'
                    )
                    logger.info(f"Successfully saved predictions for {model_type}")
                    
                except Exception as e:
                    logger.error(f"Error processing {model_type}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in run_predictions: {str(e)}")
            raise

if __name__ == "__main__":
    predictor = StockPredictor()
    predictor.run_predictions()  # Run all models by default
    # Or run specific models:
    # predictor.run_predictions(['xgboost', 'lightgbm'])
