import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
import json
import logging
import warnings
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress warnings
warnings.filterwarnings('ignore')

class StreamlitHandler(logging.Handler):
    def __init__(self, write_func):
        super().__init__()
        self.write_func = write_func

    def emit(self, record):
        log_entry = self.format(record)
        self.write_func(log_entry)

class StockPredictor:
    def __init__(self, bucket):
        self.bucket = bucket
        self.price_scaler = MinMaxScaler()  # For scaling close prices
        self.feature_scaler = StandardScaler()  # For scaling features
        self.time_steps = 30  # Increased for better temporal patterns
        self.logger = self._setup_logger()
        
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

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def _calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Price moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Volatility indicators
        df['Daily_Return'] = df['close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume indicators
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        df['Volume_StdDev'] = df['volume'].rolling(window=20).std()
        
        return df

    def _prepare_tree_based_data(self, data, train_size=0.8):
        """Prepare data for tree-based models with enhanced feature engineering"""
        try:
            self.logger.info("Preparing data for tree-based models...")
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)
            
            # Add price differences
            df['Price_Diff'] = df['close'].diff()
            df['Price_Diff_Pct'] = df['close'].pct_change()
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Split features and target
            feature_cols = [col for col in df.columns if col not in ['close']]
            X = df[feature_cols]
            y = df['close']
            
            # Train-test split
            train_size = int(len(df) * train_size)
            X_train = X[:train_size]
            X_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, df.index[train_size:], feature_cols
            
        except Exception as e:
            self.logger.error(f"Error in tree-based data preparation: {str(e)}")
            raise

    def _prepare_lstm_data(self, data, train_size=0.8):
        """Prepare data for LSTM with enhanced sequence creation"""
        try:
            self.logger.info("Preparing data for LSTM...")
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(data)
            df.dropna(inplace=True)
            
            # Scale close price separately
            price_scaled = self.price_scaler.fit_transform(df[['close']])
            df['close_scaled'] = price_scaled
            
            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['close']]
            X = df[feature_cols]
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Create sequences
            def create_sequences(X, y):
                X_seq, y_seq = [], []
                for i in range(len(X) - self.time_steps):
                    X_seq.append(X[i:(i + self.time_steps)])
                    y_seq.append(y[i + self.time_steps])
                return np.array(X_seq), np.array(y_seq)
            
            # Split into train and test
            train_size = int(len(df) * train_size)
            X_train = X_scaled[:train_size]
            X_test = X_scaled[train_size:]
            y_train = df['close_scaled'][:train_size]
            y_test = df['close'][train_size:]  # Keep original prices for evaluation
            
            # Create sequences
            X_train_seq, y_train_seq = create_sequences(X_train, y_train)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test)
            
            return X_train_seq, X_test_seq, y_train_seq, y_test_seq[self.time_steps:], df.index[train_size + self.time_steps:], feature_cols

        except Exception as e:
            self.logger.error(f"Error in LSTM data preparation: {str(e)}")
            raise

    def train_lstm(self, X_train, y_train):
        """Train LSTM model with improved architecture"""
        try:
            self.logger.info("Training LSTM model...")
            
            params = {
                "lstm_units": [128, 64, 32],
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 32,
                "patience": 10
            }
            
            model = Sequential([
                # First LSTM layer
                LSTM(params["lstm_units"][0], 
                     return_sequences=True, 
                     input_shape=(X_train.shape[1], X_train.shape[2])),
                BatchNormalization(),
                Dropout(params["dropout_rate"]),
                
                # Second LSTM layer
                LSTM(params["lstm_units"][1], 
                     return_sequences=True),
                BatchNormalization(),
                Dropout(params["dropout_rate"]),
                
                # Third LSTM layer
                LSTM(params["lstm_units"][2]),
                BatchNormalization(),
                Dropout(params["dropout_rate"]),
                
                # Dense layers
                Dense(16, activation='relu'),
                BatchNormalization(),
                Dense(1)
            ])
            
            optimizer = Adam(learning_rate=params["learning_rate"])
            model.compile(optimizer=optimizer, loss='mse')
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=params["patience"],
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
            
            # Train model
            model.fit(
                X_train, y_train,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                validation_split=0.1,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            return model, params
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise

    def predict_model(self, model, X_test, model_type):
        """Make predictions with inverse scaling for LSTM"""
        try:
            self.logger.info(f"Making predictions with {model_type} model...")
            
            if model_type == 'lstm':
                # Get scaled predictions
                scaled_pred = model.predict(X_test, verbose=0)
                # Inverse transform to get actual prices
                predictions = self.price_scaler.inverse_transform(scaled_pred)
                return predictions.flatten()
            else:
                return model.predict(X_test)
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    # [Rest of the methods remain the same: train_xgboost, train_decision_tree, train_lightgbm, 
    #  save_predictions, run_predictions, etc.]

# Streamlit interface setup
st.title("🔄 Enhanced Stock Prediction Service")

# Add instructions and run button
st.markdown("""
### Improvements Made:
1. Enhanced LSTM architecture with batch normalization
2. Improved feature engineering
3. Separate scalers for prices and features
4. Advanced technical indicators
5. Increased sequence length for better pattern recognition
6. Learning rate scheduling
7. Improved validation strategy
""")

if st.button("🚀 Run Enhanced Prediction Pipeline"):
    # Initialize and run predictor
    try:
        predictor = StockPredictor(bucket)
        predictor.run_predictions()
    except Exception as e:
        st.error(f"Error: {str(e)}")
