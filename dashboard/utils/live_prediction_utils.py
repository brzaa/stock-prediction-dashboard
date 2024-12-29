import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_rsi(data, periods=14):
    """Calculate RSI using pandas"""
    delta = data.diff()
    gain = (delta.clip(lower=0)).rolling(window=periods).mean()
    loss = (-delta.clip(upper=0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD using pandas"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def fetch_latest_data():
    """Fetch the latest stock data"""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = yf.download('MASB.JK', start='2015-01-01', end=end_date)
        
        if data.empty:
            raise ValueError("No data fetched from yfinance")
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        data.columns = data.columns.str.lower()
        data.columns = data.columns.str.replace('_masb.jk', '', regex=False)
        
        data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def calculate_features(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Technical indicators using pandas
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['signal'] = calculate_macd(df['close'])
    
    # Moving averages
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    
    # Momentum and volatility
    df['momentum'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=21).std()
    
    # Price-based features
    df['price_change'] = df['close'].diff()
    df['returns'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    
    return df

class LivePredictionPipeline:
    def __init__(self, gcs_bucket="mlops-brza"):
        self.bucket_name = gcs_bucket
        self.client = storage.Client()
        self.bucket = self.client.bucket(gcs_bucket)
        self.models = {
            'xgboost': self.train_xgboost,
            'lightgbm': self.train_lightgbm,
            'decision_tree': self.train_decision_tree,
            'lstm': self.train_lstm
        }
        
    def detect_drift(self, old_data, new_data):
        """Detect data drift using KS test"""
        drift_detected = False
        drifted_features = []
        
        for column in old_data.columns:
            _, p_value = stats.ks_2samp(old_data[column], new_data[column])
            if p_value < 0.05:
                drift_detected = True
                drifted_features.append({
                    'feature': column,
                    'p_value': float(p_value)
                })
                
        return drift_detected, drifted_features
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 500,
            "subsample": 0.8
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model, params
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        params = {
            "objective": "regression",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 500
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return model, params
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        """Train Decision Tree model"""
        params = {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10
        }
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        return model, params
    
    def train_lstm(self, X_train, y_train, X_test, y_test):
        """Train LSTM model"""
        # Reshape data for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        params = {
            "lstm_units": 50,
            "dropout": 0.2,
            "epochs": 50,
            "batch_size": 32
        }
        
        model = Sequential([
            LSTM(params["lstm_units"], input_shape=(1, X_train.shape[2])),
            Dropout(params["dropout"]),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=params["epochs"], 
                 batch_size=params["batch_size"], verbose=0)
        
        return model, params
    
    def get_predictions(self, model_type):
        """Get live predictions for specified model"""
        try:
            # Fetch and prepare data
            data = fetch_latest_data()
            data = calculate_features(data)
            
            # Remove NaN values
            data = data.dropna()
            
            # Split into train/test
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Prepare features
            feature_cols = ['rsi', 'macd', 'ma7', 'ma21', 'momentum', 'volatility', 
                          'price_change', 'returns', 'high_low_ratio', 'volume']
            X_train = train_data[feature_cols]
            y_train = train_data['close']
            X_test = test_data[feature_cols]
            y_test = test_data['close']
            
            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Check for drift
            drift_detected, drifted_features = self.detect_drift(
                pd.DataFrame(X_train_scaled, columns=feature_cols),
                pd.DataFrame(X_test_scaled, columns=feature_cols)
            )
            
            # Train model
            model, params = self.models[model_type](
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            # Make predictions
            if model_type == 'lstm':
                X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                predictions = model.predict(X_test_reshaped).flatten()
            else:
                predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'mse': float(mean_squared_error(y_test, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                'r2': float(r2_score(y_test, predictions)),
                'mae': float(mean_absolute_error(y_test, predictions))
            }
            
            # Prepare results
            results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': model_type,
                'predictions': predictions.tolist(),
                'actual_values': y_test.tolist(),
                'dates': test_data.index.strftime('%Y-%m-%d').tolist(),
                'metrics': metrics,
                'parameters': params,
                'drift_detected': drift_detected,
                'drifted_features': drifted_features,
                'feature_names': feature_cols,
                'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
            }
            
            # Save results to GCS
            blob = self.bucket.blob(f'live_predictions/{model_type}/latest.json')
            blob.upload_from_string(
                json.dumps(results),
                content_type='application/json'
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in live predictions: {str(e)}")
            raise
