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
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, bucket_name="mlops-brza"):
        """Initialize the StockPredictor with GCS bucket configuration"""
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.scaler = StandardScaler()
        self.time_steps = 5
        
        # Verify GCS connection
        self._verify_gcs_connection()
        
        # Model configurations
        self.data_prep_methods = {
            'xgboost': self._prepare_tree_based_data,
            'decision_tree': self._prepare_tree_based_data,
            'lightgbm': self._prepare_tree_based_data,
            'lstm': self._prepare_lstm_data
        }
        
        self.models = {
            'xgboost': self.train_xgboost,
            'decision_tree': self.train_decision_tree,
            'lightgbm': self.train_lightgbm,
            'lstm': self.train_lstm
        }

    def _verify_gcs_connection(self):
        """Verify connection to GCS and required folder structure"""
        try:
            if not self.bucket.exists():
                raise Exception(f"Bucket {self.bucket_name} not found")
            
            required_folders = [
                'stock_data/',
                'live_predictions/xgboost/',
                'live_predictions/lstm/',
                'live_predictions/decision_tree/',
                'live_predictions/lightgbm/'
            ]
            
            for folder in required_folders:
                blob = self.bucket.blob(folder)
                if not blob.exists():
                    logger.info(f"Creating folder: {folder}")
                    blob.upload_from_string('')
            
            logger.info("GCS connection and folder structure verified")
        except Exception as e:
            logger.error(f"GCS verification failed: {str(e)}")
            raise

    def fetch_stock_data(self):
        """Fetch stock data from GCS"""
        try:
            logger.info("Fetching stock data from GCS")
            blob = self.bucket.blob('stock_data/MASB_latest.csv')
            if not blob.exists():
                raise FileNotFoundError("Stock data file not found in GCS")
            
            local_file = '/tmp/MASB_latest.csv'
            blob.download_to_filename(local_file)
            data = pd.read_csv(local_file, parse_dates=['date'])
            data.set_index('date', inplace=True)
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def _prepare_tree_based_data(self, data, train_size=0.8):
        """Prepare data for tree-based models"""
        try:
            data['SMA_5'] = data['close'].rolling(window=5).mean()
            data['SMA_20'] = data['close'].rolling(window=20).mean()
            data['RSI'] = self._calculate_rsi(data['close'])
            data.dropna(inplace=True)
            
            train_size = int(len(data) * train_size)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            feature_cols = [col for col in data.columns if col != 'close']
            X_train = train_data[feature_cols]
            y_train = train_data['close']
            X_test = test_data[feature_cols]
            y_test = test_data['close']
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, test_data.index, feature_cols
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def _prepare_lstm_data(self, data, train_size=0.8):
        """Prepare data specifically for LSTM"""
        try:
            X_train_scaled, X_test_scaled, y_train, y_test, test_dates, feature_cols = self._prepare_tree_based_data(data, train_size)
            
            def create_sequences(X, y):
                Xs, ys = [], []
                for i in range(len(X) - self.time_steps):
                    Xs.append(X[i:(i + self.time_steps)])
                    ys.append(y.iloc[i + self.time_steps] if isinstance(y, pd.Series) else y[i + self.time_steps])
                return np.array(Xs), np.array(ys)
            
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)
            
            return X_train_seq, X_test_seq, y_train_seq, y_test_seq, test_dates[self.time_steps:], feature_cols
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {str(e)}")
            raise

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        params = {"objective": "reg:squarederror", "max_depth": 5, "learning_rate": 0.01, "n_estimators": 500}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model, params

    def train_decision_tree(self, X_train, y_train):
        """Train Decision Tree model"""
        params = {"max_depth": 10, "min_samples_split": 20, "min_samples_leaf": 10, "random_state": 42}
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        return model, params

    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        params = {"objective": "regression", "max_depth": 7, "learning_rate": 0.05, "n_estimators": 500, "num_leaves": 31}
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return model, params

    def train_lstm(self, X_train, y_train):
        """Train LSTM model"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        params = {"lstm_units_1": 64, "lstm_units_2": 32, "dropout_rate": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 32}
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params["lstm_units_1"], return_sequences=True),
            Dropout(params["dropout_rate"]),
            LSTM(params["lstm_units_2"]),
            Dropout(params["dropout_rate"]),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss='mse')
        model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=0)
        return model, params

    def predict_model(self, model, X_test, model_type):
        """Make predictions based on model type"""
        return model.predict(X_test, verbose=0).flatten() if model_type == 'lstm' else model.predict(X_test)

    def save_predictions(self, results, model_type):
        """Save prediction results to GCS"""
        blob = self.bucket.blob(f'live_predictions/{model_type}/latest.json')
        blob.upload_from_string(json.dumps(results), content_type='application/json')

    def run_predictions(self):
        """Run predictions for all models"""
        data = self.fetch_stock_data()
        for model_type in self.models:
            try:
                prepare_data = self.data_prep_methods[model_type]
                if model_type == 'lstm':
                    X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_data(data)
                else:
                    X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_data(data)
                
                model, params = self.models[model_type](X_train, y_train)
                predictions = self.predict_model(model, X_test, model_type)
                metrics = {'mse': mean_squared_error(y_test, predictions), 'rmse': np.sqrt(mean_squared_error(y_test, predictions)), 'r2': r2_score(y_test, predictions), 'mae': mean_absolute_error(y_test, predictions)}
                results = {'timestamp': datetime.now().isoformat(), 'model_type': model_type, 'predictions': predictions.tolist(), 'actual_values': y_test.tolist(), 'metrics': metrics, 'parameters': params}
                self.save_predictions(results, model_type)
                logger.info(f"Predictions saved for {model_type}")
            except Exception as e:
                logger.error(f"Error processing {model_type}: {str(e)}")

if __name__ == "__main__":
    predictor = StockPredictor()
    predictor.run_predictions()
