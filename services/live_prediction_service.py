import os
import pandas as pd
import numpy as np
from google.cloud import storage
import logging
import schedule
import time
from datetime import datetime
from typing import Dict, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
import yfinance as yf
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
            'lightgbm': self.train_lightgbm,
            'decision_tree': self.train_decision_tree
        }
    
    def fetch_stock_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            logger.info("Fetching stock data from Yahoo Finance.")
            end_date = datetime.now().strftime('%Y-%m-%d')
            data = yf.download('MASB.JK', start='2015-01-01', end=end_date, progress=False)
            
            if data.empty:
                logger.error("No stock data fetched. Please check the ticker symbol or network.")
                raise ValueError("No stock data fetched.")
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]
            data.columns = data.columns.str.lower()
            
            logger.info("Stock data fetched successfully.")
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        logger.info("Calculating technical indicators.")
        df = data.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'std_{window}'] = df['close'].rolling(window=window).std()
        
        df['momentum'] = df['close'] - df['close'].shift(5)
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        logger.info("Technical indicators calculated successfully.")
        return df

    def prepare_data(self, data: pd.DataFrame):
        """Prepare data for training"""
        logger.info("Preparing data for training.")
        data = self.calculate_features(data)
        data = data.dropna()
        
        feature_cols = [col for col in data.columns if col not in ['close']]
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        X_train = train_data[feature_cols]
        y_train = train_data['close']
        X_test = test_data[feature_cols]
        y_test = test_data['close']
        
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

    def train_lightgbm(self, X_train, y_train):
        """Train a LightGBM model"""
        params = {
            "objective": "regression",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 500
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("LightGBM model trained successfully.")
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

    def detect_drift(self, X_train, X_test):
        """Detect data drift using KS test"""
        logger.info("Detecting data drift.")
        drift_detected = False
        drifted_features = []
        
        for i in range(X_train.shape[1]):
            _, p_value = stats.ks_2samp(X_train[:, i], X_test[:, i])
            if p_value < 0.05:
                drift_detected = True
                drifted_features.append({
                    'feature_idx': i,
                    'p_value': float(p_value)
                })
        
        logger.info(f"Data drift detection completed. Drift detected: {drift_detected}.")
        return drift_detected, drifted_features

    def run_predictions(self):
        """Run predictions for all models."""
        logger.info("Starting predictions for all models.")
        try:
            # Fetch and prepare stock data
            data = self.fetch_stock_data()
            X_train, X_test, y_train, y_test, test_dates, feature_cols = self.prepare_data(data)

            # Detect data drift
            drift_detected, drifted_features = self.detect_drift(X_train, X_test)

            # Train models and save predictions
            for model_type, train_func in self.models.items():
                try:
                    model, params = train_func(X_train, y_train)
                    predictions = model.predict(X_test)

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
                        'drift_detected': drift_detected,
                        'drifted_features': drifted_features,
                        'feature_names': feature_cols
                    }

                    logger.info(f"Saving predictions for {model_type}: {json.dumps(results, indent=4)}")
                    blob = self.bucket.blob(f'live_predictions/{model_type}/latest.json')
                    blob.upload_from_string(json.dumps(results), content_type='application/json')
                    logger.info(f"Predictions saved for {model_type}.")
                except Exception as e:
                    logger.error(f"Error in {model_type} predictions: {str(e)}")
        except Exception as e:
            logger.error(f"Error in run_predictions: {str(e)}")

if __name__ == "__main__":
    predictor = StockPredictor()
    
    def job():
        logger.info("Starting scheduled prediction run.")
        predictor.run_predictions()
        logger.info("Scheduled prediction run completed.")
    
    job()  # Run immediately
    schedule.every().day.at("08:00").do(job)
    
    while True:
        try:
            schedule.run_pending()
            logger.info(f"Waiting for next scheduled run at {schedule.next_run()}.")
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Service stopped manually.")
            break
