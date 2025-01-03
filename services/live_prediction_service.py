# TensorFlow configuration (must be first)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Standard imports
import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, BatchNormalization,
    Input, LayerNormalization, Activation, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
import logging
import sys
import warnings
from datetime import datetime
import plotly.graph_objs as go
import uuid

# Memory cleanup function
def clean_memory():
    """Clean up TensorFlow memory"""
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

# Configure page
st.set_page_config(
    page_title="Live Stock Prediction Service",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Custom Streamlit logging handler
class StreamlitHandler(logging.Handler):
    """Custom logging handler to display logs in Streamlit."""
    def __init__(self, write_func):
        super().__init__()
        self.write_func = write_func

    def emit(self, record):
        log_entry = self.format(record)
        self.write_func(log_entry)

def initialize_logging():
    """Initialize logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create streamlit handler
    log_placeholder = st.empty()
    streamlit_handler = StreamlitHandler(
        lambda msg: log_placeholder.text(log_placeholder.text() + msg + "\n")
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    streamlit_handler.setFormatter(formatter)
    logger.addHandler(streamlit_handler)
    
    return logger, log_placeholder

# Initialize logger
logger, log_placeholder = initialize_logging()

@st.cache_resource
def get_gcs_client():
    """Initialize GCS client with credentials"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return storage.Client(credentials=credentials, project=credentials.project_id)
    except Exception as e:
        st.error(f"Error initializing GCS client: {str(e)}")
        return None

class StockPredictor:
    def __init__(self, bucket):
        """Initialize the StockPredictor with GCS bucket configuration"""
        logger.info("Initializing StockPredictor...")
        self.bucket = bucket
        self.scaler = MinMaxScaler()
        self.time_steps = 20
        self.run_id = str(uuid.uuid4())
        
        # Clean memory before initializing
        clean_memory()
        
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
            logger.info(f"Verifying bucket '{self.bucket.name}' exists...")
            if not self.bucket.exists():
                raise Exception(f"Bucket '{self.bucket.name}' not found.")

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

            logger.info("GCS connection and folder structure verified.")
        except Exception as e:
            logger.error(f"GCS verification failed: {str(e)}")
            raise

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def fetch_stock_data(self):
        """Fetch stock data from GCS"""
        try:
            logger.info("Fetching stock data from GCS...")
            blob = self.bucket.blob('stock_data/MASB_latest.csv')

            if not blob.exists():
                raise FileNotFoundError("Stock data file not found in GCS.")

            # Download and read data
            content = blob.download_as_string()
            data = pd.read_csv(
                pd.io.common.BytesIO(content),
                parse_dates=['date']
            )
            data.set_index('date', inplace=True)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def _prepare_tree_based_data(self, data, train_size=0.8):
        """Prepare data for tree-based models with enhanced feature engineering"""
        try:
            logger.info("Preparing data for tree-based models...")
            
            # Technical indicators
            data['SMA_5'] = data['close'].rolling(window=5).mean()
            data['SMA_20'] = data['close'].rolling(window=20).mean()
            data['RSI'] = self._calculate_rsi(data['close'])
            data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            data['BB_Middle'] = data['close'].rolling(window=20).mean()
            data['BB_Std'] = data['close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
            data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])
            
            # Additional Features
            data['Price_ROC'] = data['close'].pct_change(periods=5)
            data['Volume_ROC'] = data['volume'].pct_change(periods=5)
            data['Momentum'] = data['close'] - data['close'].shift(4)
            data['Volatility'] = data['close'].rolling(window=10).std()
            
            # Drop NaN values
            data.dropna(inplace=True)
            
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
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data preparation completed. Training set shape: {X_train_scaled.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test, test_data.index, feature_cols
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def _prepare_lstm_data(self, data, train_size=0.8):
        """Prepare data for LSTM with enhanced feature engineering"""
        try:
            logger.info("Preparing data for LSTM...")
            
            # Get base features and add LSTM-specific ones
            data['Volume_MA5'] = data['volume'].rolling(window=5).mean()
            data['Volume_MA20'] = data['volume'].rolling(window=20).mean()
            data['Price_ROC'] = data['close'].pct_change(periods=5)
            data['Volume_ROC'] = data['volume'].pct_change(periods=5)
            data['Momentum'] = data['close'] - data['close'].shift(4)
            data['Volatility'] = data['close'].rolling(window=10).std()
            
            # Get scaled data from tree-based preparation
            X_train_scaled, X_test_scaled, y_train, y_test, test_dates, feature_cols = (
                self._prepare_tree_based_data(data, train_size)
            )
            
            # Create sequences
            def create_sequences(X, y):
                Xs, ys = [], []
                for i in range(len(X) - self.time_steps):
                    sequence = X[i:(i + self.time_steps)]
                    target = y.iloc[i + self.time_steps] if isinstance(y, pd.Series) else y[i + self.time_steps]
                    Xs.append(sequence)
                    ys.append(target)
                return np.array(Xs), np.array(ys)
            
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)
            
            logger.info(f"LSTM data preparation completed. Sequence shape: {X_train_seq.shape}")
            return X_train_seq, X_test_seq, y_train_seq, y_test_seq, test_dates[self.time_steps:], feature_cols
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {str(e)}")
            raise

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with optimized parameters"""
        logger.info("Training XGBoost model...")
        params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("XGBoost training completed.")
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
        logger.info("Decision Tree training completed.")
        return model, params

    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        logger.info("Training LightGBM model...")
        params = {
            "objective": "regression",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "num_leaves": 31,
            "random_state": 42
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("LightGBM training completed.")
        return model, params

    def train_lstm(self, X_train, y_train):
        """Train enhanced LSTM model with advanced architecture"""
        try:
            logger.info("Training enhanced LSTM model...")
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            # Clean memory before training
            clean_memory()
            
            params = {
                "lstm_units": [256, 128, 64],
                "dense_units": [128, 64, 32],
                "dropout_rates": [0.3, 0.3, 0.3],
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 32,
                "patience": 15
            }

            # Build model
            input_layer = Input(shape=input_shape)
            
            # First LSTM block
            x = Bidirectional(LSTM(params["lstm_units"][0], 
                                return_sequences=True,
                                kernel_initializer='he_normal'))(input_layer)
            x = LayerNormalization()(x)
            x = Dropout(params["dropout_rates"][0])(x)
            
            # Attention mechanism
            attention = Dense(1, activation='tanh')(x)
            attention = Activation('softmax')(attention)
            x = Multiply()([x, attention])
            
            # Second LSTM block
            x = Bidirectional(LSTM(params["lstm_units"][1], 
                                return_sequences=True,
                                kernel_initializer='he_normal'))(x)
            x = LayerNormalization()(x)
            x = Dropout(params["dropout_rates"][1])(x)
            
            # Final LSTM block
            x = Bidirectional(LSTM(params["lstm_units"][2],
                                kernel_initializer='he_normal'))(x)
            x = LayerNormalization()(x)
            x = Dropout(params["dropout_rates"][2])(x)
            
            # Dense layers
            for units in params["dense_units"]:
                x = Dense(units, activation='swish')(x)
                x = LayerNormalization()(x)
                x = Dropout(0.2)(x)
            
            output_layer = Dense(1, activation='linear')(x)
            
            model = Model(inputs=input_layer, outputs=output_layer)
            
            # Optimizer and compilation
            optimizer = Adam(
                learning_rate=params["learning_rate"],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=True
            )
            
            def custom_loss(y_true, y_pred):
                mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
                huber = tf.keras.losses.huber(y_true, y_pred, delta=1.0)
                return 0.7 * mse + 0.3 * huber
            
            model.compile(
                optimizer=optimizer,
                loss=custom_loss,
                metrics=['mae', 'mse']
            )

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=params["patience"],
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Add noise for regularization
            noise_factor = 0.01
            X_train_noisy = X_train + noise_factor * np.random.normal(0, 1, X_train.shape)
            
            # Split validation set
            split = int(0.8 * len(X_train))
            X_tr, X_val = X_train_noisy[:split], X_train[split:]
            y_tr, y_val = y_train[:split], y_train[split:]
            
            # Train model
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("LSTM training completed successfully.")
            
            # Add training history to parameters
            params["training_history"] = {
                "loss": float(history.history["loss"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "final_lr": float(K.eval(model.optimizer.lr))
            }
            
            return model, params
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            raise

    def predict_model(self, model, X_test, model_type):
        """Make predictions with specified model"""
        logger.info(f"Making predictions with {model_type} model...")
        try:
            if model_type == 'lstm':
                predictions = model.predict(X_test, verbose=0).flatten()
            else:
                predictions = model.predict(X_test)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def save_predictions(self, results, model_type):
        """Save prediction results to GCS"""
        try:
            logger.info(f"Saving predictions for {model_type} model...")
            
            # Add run_id and timestamp to results
            results['run_id'] = self.run_id
            results['timestamp'] = datetime.now().isoformat()
            
            # Save to live predictions
            live_blob = self.bucket.blob(f'live_predictions/{model_type}/latest.json')
            
            # Ensure all numpy types are converted to Python native types
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Convert results to JSON-serializable format
            processed_results = {
                key: convert_to_native(value)
                for key, value in results.items()
            }
            
            # Save to both locations
            live_blob.upload_from_string(
                json.dumps(processed_results, indent=2),
                content_type='application/json'
            )
            
            # Save to model outputs with run_id
            output_blob = self.bucket.blob(f'model_outputs/{model_type}/{self.run_id}/results.json')
            output_blob.upload_from_string(
                json.dumps(processed_results, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"Predictions saved for {model_type} model.")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {str(e)}")
            raise

    def run_predictions(self):
        """Run predictions for all models"""
        try:
            logger.info("Starting prediction pipeline...")
            data = self.fetch_stock_data()

            for model_type in self.models:
                try:
                    logger.info(f"\n{'='*50}\nProcessing {model_type} model\n{'='*50}")
                    prepare_data = self.data_prep_methods[model_type]

                    # Prepare data
                    if model_type == 'lstm':
                        X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_data(data)
                        # Clean memory before LSTM training
                        clean_memory()
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
                        'mae': float(mean_absolute_error(y_test, predictions)),
                        'mape': float(mean_absolute_percentage_error(y_test, predictions))
                    }

                    # Prepare results
                    results = {
                        'model_type': model_type,
                        'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                        'actual_values': y_test.tolist() if isinstance(y_test, np.ndarray) else y_test,
                        'dates': [str(date) for date in test_dates],
                        'metrics': metrics,
                        'parameters': params
                    }

                    # Save results
                    self.save_predictions(results, model_type)

                    # Display metrics in Streamlit
                    st.subheader(f"📊 {model_type.capitalize()} Model Metrics")
                    st.json(metrics)

                    # Visualize predictions
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=test_dates,
                        y=y_test,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=test_dates,
                        y=predictions,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red', width=2)
                    ))
                    fig.update_layout(
                        title=f"Actual vs Predicted for {model_type.capitalize()} Model",
                        xaxis_title='Date',
                        yaxis_title='Price',
                        legend=dict(x=0, y=1),
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    logger.info(f"{model_type.capitalize()} Model Metrics: {metrics}")

                    # Clean up after each model
                    if model_type == 'lstm':
                        clean_memory()

                except Exception as e:
                    logger.error(f"Error processing {model_type} model: {str(e)}")
                    st.error(f"❌ Error processing {model_type} model. Check the logs for more details.")
                    continue

            logger.info("Prediction pipeline completed successfully.")
            st.success("✅ Prediction pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Prediction pipeline failed: {str(e)}")
            st.error("❌ Prediction pipeline failed. Check the logs for more details.")
            raise

def main():
    st.title("🟢 Live Stock Prediction Service")
    st.markdown("""
    This service runs predictions using multiple models:
    - XGBoost
    - Decision Tree
    - LightGBM
    - LSTM
    
    Each model's predictions and metrics will be saved to Google Cloud Storage.
    """)
    
    try:
        # Get GCS client
        client = get_gcs_client()
        if not client:
            st.error("Failed to initialize Google Cloud client")
            st.stop()
            
        # Get bucket
        bucket = client.bucket("mlops-brza")
        
        if st.button("🚀 Run Prediction Pipeline"):
            with st.spinner('Running predictions...'):
                predictor = StockPredictor(bucket)
                predictor.run_predictions()
                
    except Exception as e:
        st.error(f"Fatal error occurred: {str(e)}")
        logger.error(f"Fatal error occurred: {str(e)}")
    finally:
        clean_memory()

if __name__ == "__main__":
    main()
