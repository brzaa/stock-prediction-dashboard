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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Import callbacks
import json
import warnings
from google.colab import auth
warnings.filterwarnings('ignore')

# Configure logging to show more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, bucket_name="mlops-brza"):
        """Initialize the StockPredictor with GCS bucket configuration"""
        logger.info("Initializing StockPredictor...")
        self.bucket_name = bucket_name
        
        # Authenticate with Google Cloud
        try:
            logger.info("Authenticating with Google Cloud...")
            auth.authenticate_user()
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
            logger.info("Authentication successful")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
        
        self.scaler = StandardScaler()
        self.time_steps = 1  # Single timestep for LSTM (no rolling window)
        
        # Verify GCS connection
        self._verify_gcs_connection()
        
        # Model configurations with reduced complexity for testing
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
            logger.info(f"Verifying bucket {self.bucket_name} exists...")
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
                logger.info(f"Checking folder: {folder}")
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
            
            # Check if file exists
            logger.info("Checking if stock data file exists...")
            if not blob.exists():
                raise FileNotFoundError("Stock data file not found in GCS at stock_data/MASB_latest.csv")
            
            logger.info("Downloading stock data file...")
            local_file = '/tmp/MASB_latest.csv'
            blob.download_to_filename(local_file)
            
            logger.info("Reading stock data into DataFrame...")
            data = pd.read_csv(local_file, parse_dates=['date'])
            data.set_index('date', inplace=True)
            
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _prepare_tree_based_data(self, data, train_size=0.8):
        """Prepare data for tree-based models"""
        try:
            logger.info("Preparing data for tree-based models...")
            
            # Calculate features
            logger.info("Calculating technical indicators...")
            data['SMA_5'] = data['close'].rolling(window=5).mean()
            data['SMA_20'] = data['close'].rolling(window=20).mean()
            data['RSI'] = self._calculate_rsi(data['close'])
            
            # Drop NaN values
            original_length = len(data)
            data.dropna(inplace=True)
            logger.info(f"Dropped {original_length - len(data)} rows with NaN values")
            
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
            logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data preparation completed. Training set shape: {X_train_scaled.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test, test_data.index, feature_cols
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def _prepare_lstm_data(self, data, train_size=0.8):
        """Prepare data specifically for LSTM (no rolling window)"""
        try:
            logger.info("Preparing data for LSTM (no rolling window)...")
            X_train_scaled, X_test_scaled, y_train, y_test, test_dates, feature_cols = self._prepare_tree_based_data(data, train_size)
            
            # Reshape for LSTM (samples, timesteps, features)
            X_train_seq = X_train_scaled.reshape(X_train_scaled.shape[0], self.time_steps, X_train_scaled.shape[1])
            X_test_seq = X_test_scaled.reshape(X_test_scaled.shape[0], self.time_steps, X_test_scaled.shape[1])
            
            logger.info(f"LSTM data preparation completed. Training sequence shape: {X_train_seq.shape}")
            return X_train_seq, X_test_seq, y_train, y_test, test_dates, feature_cols
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {str(e)}")
            raise

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        # Reduced number of estimators for testing
        params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 100  # Reduced from 500
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("XGBoost training completed")
        return model, params

    def train_decision_tree(self, X_train, y_train):
        """Train Decision Tree model"""
        logger.info("Training Decision Tree model...")
        params = {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42
        }
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("Decision Tree training completed")
        return model, params

    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        logger.info("Training LightGBM model...")
        # Reduced number of estimators for testing
        params = {
            "objective": "regression",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 100,  # Reduced from 500
            "num_leaves": 31
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("LightGBM training completed")
        return model, params

    def train_lstm(self, X_train, y_train):
        """Train LSTM model with improved configuration"""
        logger.info("Training LSTM model with enhanced configuration...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        params = {
            "lstm_units_1": 128,  # Increased units
            "lstm_units_2": 64,   # Additional layer
            "dropout_rate": 0.3,  # Increased dropout
            "learning_rate": 0.0005,  # Adjusted learning rate
            "epochs": 50,  # Increased epochs
            "batch_size": 32,  # Reduced batch size
            "validation_split": 0.1  # Validation split
        }
        
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params["lstm_units_1"], return_sequences=True, recurrent_dropout=0.2, kernel_regularizer='l2'),
            Dropout(params["dropout_rate"]),
            LSTM(params["lstm_units_2"], recurrent_dropout=0.2, kernel_regularizer='l2'),
            Dropout(params["dropout_rate"]),
            Dense(32, activation='relu'),  # Additional dense layer
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params["learning_rate"]),
            loss='huber'  # Use Huber loss for robustness to outliers
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        model.fit(
            X_train, y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_split=params["validation_split"],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM training completed")
        return model, params

    def predict_model(self, model, X_test, model_type):
        """Make predictions based on model type"""
        logger.info(f"Making predictions with {model_type} model...")
        return model.predict(X_test, verbose=0).flatten() if model_type == 'lstm' else model.predict(X_test)

    def save_predictions(self, results, model_type):
        """Save prediction results to GCS"""
        logger.info(f"Saving predictions for {model_type}...")
        blob = self.bucket.blob(f'live_predictions/{model_type}/latest.json')
        blob.upload_from_string(json.dumps(results), content_type='application/json')
        logger.info(f"Predictions saved for {model_type}")

    def run_predictions(self):
        """Run predictions for all models"""
        logger.info("Starting prediction pipeline...")
        data = self.fetch_stock_data()
        
        for model_type in self.models:
            try:
                logger.info(f"\n{'='*50}\nProcessing {model_type} model\n{'='*50}")
                prepare_data = self.data_prep_methods[model_type]
                
                # Prepare data
                X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_data(data)
                
                # Train model and make predictions
                model, params = self.models[model_type](X_train, y_train)
                predictions = self.predict_model(model, X_test, model_type)
                
                # Calculate metrics
                metrics = {
                    'mse': mean_squared_error(y_test, predictions),
                    'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                    'r2': r2_score(y_test, predictions),
                    'mae': mean_absolute_error(y_test, predictions)
                }
                
                # Prepare results
                results = {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type,
                    'predictions': predictions.tolist(),
                    'actual_values': y_test.tolist(),
                    'metrics': metrics,
                    'parameters': params
                }
                
                # Save results
                self.save_predictions(results, model_type)
                
            except Exception as e:
                logger.error(f"Error processing {model_type}: {str(e)}")
                continue

if __name__ == "__main__":
    try:
        logger.info("="*50)
        logger.info("Starting Stock Prediction Service")
        logger.info("="*50)
        
        # Initialize and run predictor
        predictor = StockPredictor()
        predictor.run_predictions()
        
        logger.info("="*50)
        logger.info("Stock Prediction Service Completed Successfully")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Fatal error occurred: {str(e)}")
        raise
