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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import stats
import json
import warnings
from google.colab import auth

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class DataDriftManager:
    def __init__(self):
        self.untrained_data = pd.DataFrame()
        self.alpha_drift_threshold = 0.15
        self.reference_data = None
        self.new_data_buffer = pd.DataFrame()
        
    def store_untrained_data(self, data):
        self.untrained_data = pd.concat([self.untrained_data, data])
        logger.info(f"Stored {len(data)} new records. Total untrained: {len(self.untrained_data)}")
    
    def set_reference_data(self, data):
        self.reference_data = data.copy()
        logger.info("Reference data updated")
    
    def add_new_data(self, data):
        self.new_data_buffer = pd.concat([self.new_data_buffer, data])
        logger.info(f"Added {len(data)} records to buffer. Total: {len(self.new_data_buffer)}")
    
    def check_alpha_drift(self, new_data):
        """
        Checks for alpha drift by comparing statistical metrics from reference_data vs. new_data.
        Converts all numeric/boolean values to Python-native types to avoid JSON serialization errors.
        """
        if self.reference_data is None:
            return False, {}
            
        drift_metrics = {}
        significant_drift = False
        
        for column in self.reference_data.columns:
            if column in new_data.columns:
                ref_stats = self._calculate_stats(self.reference_data[column])
                new_stats = self._calculate_stats(new_data[column])
                
                # Calculate drift
                drift_val = abs(new_stats['mean'] - ref_stats['mean']) / ref_stats['std']
                # Convert to native Python types
                drift_magnitude = float(drift_val) if not pd.isna(drift_val) else 0.0
                is_significant = bool(drift_magnitude > self.alpha_drift_threshold)
                
                drift_metrics[column] = {
                    'drift_magnitude': drift_magnitude,
                    'is_significant': is_significant,
                    'ref_stats': {
                        'mean': float(ref_stats['mean']) if ref_stats['mean'] is not None else 0.0,
                        'std': float(ref_stats['std']) if ref_stats['std'] is not None else 0.0,
                        'median': float(ref_stats['median']) if ref_stats['median'] is not None else 0.0,
                        'q1': float(ref_stats['q1']) if ref_stats['q1'] is not None else 0.0,
                        'q3': float(ref_stats['q3']) if ref_stats['q3'] is not None else 0.0,
                    },
                    'new_stats': {
                        'mean': float(new_stats['mean']) if new_stats['mean'] is not None else 0.0,
                        'std': float(new_stats['std']) if new_stats['std'] is not None else 0.0,
                        'median': float(new_stats['median']) if new_stats['median'] is not None else 0.0,
                        'q1': float(new_stats['q1']) if new_stats['q1'] is not None else 0.0,
                        'q3': float(new_stats['q3']) if new_stats['q3'] is not None else 0.0,
                    }
                }
                
                if is_significant:
                    significant_drift = True
        
        return significant_drift, drift_metrics
    
    def _calculate_stats(self, series):
        return {
            'mean': series.mean(),
            'std': series.std(),
            'median': series.median(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75)
        }


class ModelVersionControl:
    def __init__(self):
        self.current_version = {
            'beta': None,
            'alpha': None,
            'production': None
        }
        self.version_history = []
        
    def promote_model(self, model_artifact, drift_analysis):
        if drift_analysis['severity'] == 'HIGH':
            new_version = self._create_beta_version(model_artifact)
            self.current_version['beta'] = new_version
            self.version_history.append(new_version)
            return 'BETA', new_version
            
        elif drift_analysis['severity'] == 'MODERATE':
            if self._is_beta_stable():
                new_version = self._promote_to_alpha()
                self.current_version['alpha'] = new_version
                self.version_history.append(new_version)
                return 'ALPHA', new_version
                
        return None, None
        
    def _create_beta_version(self, model_artifact):
        return {
            'id': f"beta_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'artifact': model_artifact,
            'status': 'beta',
            'metrics': {},
            'training_data_snapshot': None
        }
        
    def _promote_to_alpha(self):
        if self.current_version['beta'] is None:
            raise ValueError("No beta version available")
            
        alpha_version = self.current_version['beta'].copy()
        alpha_version['id'] = f"alpha_{int(time.time())}"
        alpha_version['status'] = 'alpha'
        return alpha_version
        
    def _is_beta_stable(self):
        if self.current_version['beta'] is None:
            return False
        return True


class StockPredictor:
    def __init__(self, bucket_name="mlops-brza"):
        logger.info("Initializing StockPredictor...")
        self.bucket_name = bucket_name
        
        try:
            auth.authenticate_user()
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
            logger.info("GCloud authentication successful")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
        
        self.scaler = StandardScaler()
        self.drift_manager = DataDriftManager()
        self.version_control = ModelVersionControl()
        self._verify_gcs_connection()
        
        # One helper method for each model type
        self.data_prep_methods = {
            'xgboost': self._prepare_tree_based_data,
            'decision_tree': self._prepare_tree_based_data,
            'lightgbm': self._prepare_tree_based_data,
            'lstm': self._prepare_lstm_data
        }
        
        # Training method for each model type
        self.models = {
            'xgboost': self.train_xgboost,
            'decision_tree': self.train_decision_tree,
            'lightgbm': self.train_lightgbm,
            'lstm': self.train_lstm
        }

    def _verify_gcs_connection(self):
        try:
            if not self.bucket.exists():
                raise Exception(f"Bucket {self.bucket_name} not found")
            
            required_folders = [
                'stock_data/', 'live_predictions/', 
                'drift_analysis/', 'model_events/', 'version_control/'
            ]
            
            for folder in required_folders:
                blob = self.bucket.blob(folder)
                if not blob.exists():
                    blob.upload_from_string('')
            
            logger.info("GCS structure verified")
        except Exception as e:
            logger.error(f"GCS verification failed: {str(e)}")
            raise

    def prepare_training_data(self, data, train_size=0.8):
        """
        Prepares training data, checking for alpha drift among the latest 30 days.
        If drift is detected, the new data is stored separately as untrained and only old_data is used.
        
        Returns (X_train, X_test, y_train, y_test). 
        If there's insufficient data after dropna(), returns None for all four.
        """
        cutoff_date = data.index.max() - pd.Timedelta(days=30)
        old_data = data[data.index <= cutoff_date]
        new_data = data[data.index > cutoff_date]
        
        # Add new data to the drift buffer
        self.drift_manager.add_new_data(new_data)
        
        has_drift, drift_metrics = self.drift_manager.check_alpha_drift(new_data)
        if has_drift:
            logger.warning("Alpha drift detected")
            self._handle_alpha_drift(drift_metrics)
            self.drift_manager.store_untrained_data(new_data)
            train_data = old_data
        else:
            train_data = data
        
        # Build features (rolling windows, etc.) and drop NaNs
        X = self._prepare_features(train_data)
        # Align y with X's index
        y = train_data['close'].reindex(X.index)
        
        # If the dataset ended up empty after feature engineering, abort training
        if len(X) == 0 or len(y) == 0:
            logger.error("Not enough valid rows in train_data after dropna(). Cannot train.")
            return None, None, None, None

        if has_drift:
            X_train = X[X.index <= cutoff_date]
            X_test = X[X.index > cutoff_date]
            y_train = y[y.index <= cutoff_date]
            y_test = y[y.index > cutoff_date]
        else:
            # Standard train/test split
            train_idx = int(len(X) * train_size)
            X_train, X_test = X[:train_idx], X[train_idx:]
            y_train, y_test = y[:train_idx], y[train_idx:]
        
        # Final check: if either train set is empty, skip training
        if len(X_train) == 0 or len(y_train) == 0:
            logger.error("X_train or y_train is empty. Skipping training.")
            return None, None, None, None
        
        return X_train, X_test, y_train, y_test

    def _prepare_features(self, data):
        """
        Basic feature engineering for all model types.
        """
        features = pd.DataFrame(index=data.index)
        
        # Technical indicators
        features['SMA_5'] = data['close'].rolling(window=5).mean()
        features['SMA_20'] = data['close'].rolling(window=20).mean()
        features['RSI'] = self._calculate_rsi(data['close'])
        features['MACD'] = self._calculate_macd(data['close'])
        features['BB_upper'], features['BB_lower'] = self._calculate_bollinger_bands(data['close'])
        features['Volume_SMA'] = data['volume'].rolling(window=20).mean()
        features['Price_Range'] = data['high'] - data['low']
        
        # Drop rows with NaN (particularly from rolling windows)
        features.dropna(inplace=True)
        return features

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd - macd.ewm(span=signal).mean()

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma + (std_dev * std), sma - (std_dev * std)
        
    def _prepare_tree_based_data(self, data, train_size=0.8):
        """
        Prepare data for tree-based models (XGBoost, LightGBM, Decision Tree).
        """
        try:
            logger.info("Preparing data for tree-based models...")
            
            # Calculate features
            X = self._prepare_features(data)
            y = data['close'].reindex(X.index)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid rows after feature engineering for tree-based. Returning None.")
                return None, None, None, None, None
            
            # Split data
            train_size_int = int(len(X) * train_size)
            X_train = X[:train_size_int]
            X_test = X[train_size_int:]
            y_train = y[:train_size_int]
            y_test = y[train_size_int:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data preparation completed. Training set shape: {X_train_scaled.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test, X[test_size_int:].index
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
            
    def _prepare_lstm_data(self, data, train_size=0.8, sequence_length=60):
        """
        Prepare data specifically for LSTM (creates sequences).
        """
        try:
            logger.info("Preparing data for LSTM...")
            
            # First, prepare data as if it's tree-based, to scale & align
            prepared = self._prepare_tree_based_data(data, train_size)
            if prepared is None:
                return None, None, None, None, None
            
            X_train_scaled, X_test_scaled, y_train, y_test, test_dates = prepared
            
            if X_train_scaled is None or len(X_train_scaled) == 0:
                logger.error("No valid rows for LSTM training after preparing data. Returning None.")
                return None, None, None, None, None
            
            # Create sequences for LSTM
            def create_sequences(X, y, seq_length):
                Xs, ys = [], []
                for i in range(len(X) - seq_length):
                    Xs.append(X[i:(i + seq_length)])
                    ys.append(y[i + seq_length])
                return np.array(Xs), np.array(ys)
            
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, sequence_length)
            
            logger.info(f"LSTM data preparation completed. Training sequence shape: {X_train_seq.shape}")
            return X_train_seq, X_test_seq, y_train_seq, y_test_seq, test_dates
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {str(e)}")
            raise

    def fetch_stock_data(self):
        """
        Fetch stock data from GCS and process it. 
        Also checks for drift if reference_data is already set.
        """
        try:
            logger.info("Fetching stock data from GCS")
            blob = self.bucket.blob('stock_data/MASB_latest.csv')
            
            if not blob.exists():
                raise FileNotFoundError("Stock data file not found in GCS")
            
            logger.info("Downloading stock data file...")
            local_file = '/tmp/MASB_latest.csv'
            blob.download_to_filename(local_file)
            
            logger.info("Reading stock data into DataFrame...")
            data = pd.read_csv(local_file, parse_dates=['date'])
            data.set_index('date', inplace=True)
            
            # Check for drift if reference exists
            if self.drift_manager.reference_data is not None:
                drift_detected, drift_report = self.drift_manager.check_alpha_drift(data)
                if drift_detected:
                    logger.warning("Data drift detected")
                    drift_analysis, action_taken = self.handle_data_drift(data, drift_report)
                    logger.info(f"Drift handling completed. Action: {action_taken['type']}")
            else:
                self.drift_manager.set_reference_data(data)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def _handle_alpha_drift(self, drift_metrics):
        """
        Logs drift information to GCS. 
        If enough untrained data accumulates, triggers retraining.
        """
        drift_log = {
            'timestamp': datetime.now().isoformat(),
            'drift_metrics': drift_metrics,  # Already converted to Python-native floats/bools
            'action_taken': 'separate_training'
        }
        
        drift_log_blob = self.bucket.blob('drift_analysis/alpha_drift_log.json')
        existing_logs = []
        
        if drift_log_blob.exists():
            existing_logs = json.loads(drift_log_blob.download_as_string())
        
        existing_logs.append(drift_log)
        
        # Write JSON back to GCS
        drift_log_blob.upload_from_string(json.dumps(existing_logs))
        
        # If there's enough untrained data, retrain
        if len(self.drift_manager.untrained_data) >= 1000:
            logger.info("Sufficient untrained data accumulated. Starting retraining...")
            self._trigger_retraining()
    
    def _trigger_retraining(self):
        """
        Combine reference_data with untrained_data, retrain the model, 
        and reset the untrained_data buffer.
        """
        try:
            all_data = pd.concat([
                self.drift_manager.reference_data,
                self.drift_manager.untrained_data
            ]).sort_index()
            
            X_train, X_test, y_train, y_test = self.prepare_training_data(all_data)
            if X_train is None or len(X_train) == 0:
                logger.warning("Cannot retrain model because training set is empty.")
                return None

            new_model = self.train_model(X_train, y_train)
            
            # Update reference data and clear untrained buffer
            self.drift_manager.set_reference_data(all_data)
            self.drift_manager.untrained_data = pd.DataFrame()
            
            logger.info("Model retraining completed")
            return new_model
        except Exception as e:
            logger.error(f"Retraining error: {str(e)}")
            raise

    def handle_data_drift(self, data, drift_report):
        """
        (Optional) Custom logic to handle drift. 
        For demonstration, we store the new data and 
        then return a placeholder drift_analysis & action.
        """
        drift_analysis = {'severity': 'HIGH'}
        action_taken = {'type': 'PROMOTE_BETA'}
        return drift_analysis, action_taken

    def train_model(self, X_train, y_train, model_type='xgboost'):
        """
        Utility that picks the right model training function.
        """
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        train_func = self.models[model_type]
        return train_func(X_train, y_train)

    def train_xgboost(self, X_train, y_train):
        params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 100
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model, params

    def train_decision_tree(self, X_train, y_train):
        params = {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42
        }
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        return model, params

    def train_lightgbm(self, X_train, y_train):
        params = {
            "objective": "regression",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "num_leaves": 31
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return model, params

    def train_lstm(self, X_train, y_train):
        # For LSTM, X_train should be 3D if you've already created sequences.
        params = {
            "lstm_units": 128,
            "dropout_rate": 0.3,
            "learning_rate": 0.0005,
            "epochs": 50,
            "batch_size": 32
        }
        
        model = Sequential([
            LSTM(params["lstm_units"], return_sequences=True),
            Dropout(params["dropout_rate"]),
            LSTM(params["lstm_units"]//2),
            Dropout(params["dropout_rate"]),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss='huber')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        model.fit(
            X_train, y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, params

    def evaluate_models(self, X_test, y_test, models_dict):
        """
        Evaluate a dictionary of trained models on the same test set.
        """
        evaluations = {}
        for model_name, model in models_dict.items():
            predictions = model.predict(X_test)
            evaluations[model_name] = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }
        return evaluations


if __name__ == "__main__":
    try:
        logger.info("Starting Stock Prediction Pipeline")
        predictor = StockPredictor()
        
        # Fetch and prepare data
        data = predictor.fetch_stock_data()
        X_train, X_test, y_train, y_test = predictor.prepare_training_data(data)
        
        # If we don't have a valid training set, skip the training & exit
        if X_train is None or X_test is None or y_train is None or y_test is None:
            logger.warning("Skipping training because training set is empty or None.")
        else:
            # Train models
            trained_models = {}
            for model_type in predictor.models:
                logger.info(f"Training {model_type} model...")
                model, _ = predictor.train_model(X_train, y_train, model_type)
                trained_models[model_type] = model
            
            # Evaluate models
            evaluations = predictor.evaluate_models(X_test, y_test, trained_models)
            logger.info("Model evaluations:")
            logger.info(json.dumps(evaluations, indent=2))
        
        logger.info("Stock Prediction Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
