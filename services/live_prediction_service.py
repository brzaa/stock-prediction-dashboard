# live_prediction_service.py

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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import json
import logging
import sys
import warnings
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import plotly.graph_objs as go

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging to display logs in Streamlit
class StreamlitHandler(logging.Handler):
    """Custom logging handler to display logs in Streamlit."""
    def __init__(self, write_func):
        super().__init__()
        self.write_func = write_func

    def emit(self, record):
        log_entry = self.format(record)
        self.write_func(log_entry)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a placeholder for logs
log_placeholder = st.empty()

def log_function(msg):
    """Function to append log messages to the Streamlit placeholder."""
    current_text = log_placeholder.text()
    new_text = current_text + msg + "\n"
    log_placeholder.text(new_text)

# Add the custom Streamlit handler to the logger
streamlit_handler = StreamlitHandler(log_function)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
streamlit_handler.setFormatter(formatter)
logger.addHandler(streamlit_handler)

# Streamlit App Title
st.title("🟢 Live Stock Prediction Service")

# Instructions
st.markdown("""
This application fetches stock data from Google Cloud Storage (GCS), trains multiple machine learning models, makes predictions, and saves the results back to GCS.

### Steps Performed:
1. **Authenticate** with GCS using a service account.
2. **Fetch** the latest stock data.
3. **Prepare** the data for various models.
4. **Train** models: XGBoost, Decision Tree, LightGBM, LSTM.
5. **Make Predictions** using the trained models.
6. **Save** predictions back to GCS.
7. **Display** metrics and logs.
""")

# Button to Run Prediction Pipeline
if st.button("🚀 Run Prediction Pipeline"):
    with st.spinner('Initializing Stock Prediction Service...'):
        try:
            # Read GCP service account from Streamlit Secrets
            service_account_info = st.secrets["gcp"]["service_account"]
            service_account_info = json.loads(service_account_info)

            # Authenticate with Google Cloud using service account credentials
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            client = storage.Client(credentials=credentials, project=credentials.project_id)
            bucket_name = "mlops-brza"  # Replace with your bucket name if different
            bucket = client.bucket(bucket_name)
            logger.info("Authentication with Google Cloud successful.")
            logger.info(f"Using service account: {credentials.service_account_email}")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            st.error("❌ Authentication with Google Cloud failed. Check the logs for more details.")
            st.stop()

    # Define the StockPredictor class
    class StockPredictor:
        def __init__(self, bucket):
            """Initialize the StockPredictor with GCS bucket configuration"""
            logger.info("Initializing StockPredictor...")
            self.bucket = bucket
            self.scaler = MinMaxScaler()
            self.time_steps = 20  # Increased sequence length for better temporal patterns

            # Verify GCS connection
            self._verify_gcs_connection()

            # Model configurations with optimized parameters
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
                        blob.upload_from_string('')  # Create an empty blob to represent the folder

                logger.info("GCS connection and folder structure verified.")
            except Exception as e:
                logger.error(f"GCS verification failed: {str(e)}")
                raise

        def fetch_stock_data(self):
            """Fetch stock data from GCS"""
            try:
                logger.info("Fetching stock data from GCS...")
                blob = self.bucket.blob('stock_data/MASB_latest.csv')

                # Check if file exists
                logger.info("Checking if stock data file exists...")
                if not blob.exists():
                    raise FileNotFoundError("Stock data file not found in GCS at 'stock_data/MASB_latest.csv'.")

                logger.info("Downloading stock data file...")
                local_file = '/tmp/MASB_latest.csv'
                blob.download_to_filename(local_file)
                logger.info("Download completed.")

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

                # Calculate technical indicators
                logger.info("Calculating technical indicators...")
                data['SMA_5'] = data['close'].rolling(window=5).mean()
                data['SMA_20'] = data['close'].rolling(window=20).mean()
                data['RSI'] = self._calculate_rsi(data['close'])

                # Additional technical indicators
                logger.info("Calculating additional technical indicators (MACD, Bollinger Bands)...")
                data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
                data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

                # Bollinger Bands
                data['BB_Middle'] = data['close'].rolling(window=20).mean()
                data['BB_Std'] = data['close'].rolling(window=20).std()
                data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
                data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])

                # Drop NaN values
                original_length = len(data)
                data.dropna(inplace=True)
                logger.info(f"Dropped {original_length - len(data)} rows with NaN values.")

                # Split data
                train_size = int(len(data) * train_size)
                train_data = data[:train_size]
                test_data = data[train_size:]

                feature_cols = [col for col in data.columns if col != 'close']
                X_train = train_data[feature_cols]
                y_train = train_data['close']
                X_test = test_data[feature_cols]
                y_test = test_data['close']

                # Scale features with MinMaxScaler
                logger.info("Scaling features with MinMaxScaler...")
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                logger.info(f"Data preparation completed. Training set shape: {X_train_scaled.shape}")
                return X_train_scaled, X_test_scaled, y_train, y_test, test_data.index, feature_cols
            except Exception as e:
                logger.error(f"Error in data preparation: {str(e)}")
                raise

        def _prepare_lstm_data(self, data, train_size=0.8):
            """Prepare data specifically for LSTM with enhanced feature engineering"""
            try:
                logger.info("Preparing data for LSTM with enhanced features...")
                
                # Calculate additional technical indicators
                data['Volume_MA5'] = data['volume'].rolling(window=5).mean()
                data['Volume_MA20'] = data['volume'].rolling(window=20).mean()
                data['Price_ROC'] = data['close'].pct_change(periods=5)
                data['Volume_ROC'] = data['volume'].pct_change(periods=5)
                
                # Calculate price momentum
                data['Momentum'] = data['close'] - data['close'].shift(4)
                
                # Volatility indicator
                data['Volatility'] = data['close'].rolling(window=10).std()
                
                # Get the base features from tree-based preparation
                X_train_scaled, X_test_scaled, y_train, y_test, test_dates, feature_cols = self._prepare_tree_based_data(data, train_size)
                
                logger.info("Creating sequences for LSTM...")
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

                logger.info(f"LSTM data preparation completed. Training sequence shape: {X_train_seq.shape}")
                return X_train_seq, X_test_seq, y_train_seq, y_test_seq, test_dates[self.time_steps:], feature_cols
                
            except Exception as e:
                logger.error(f"Error preparing LSTM data: {str(e)}")
                raise

        def train_xgboost(self, X_train, y_train):
            """Train XGBoost model"""
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
            """Train LSTM model with improved architecture and training strategy"""
            try:
                logger.info("Training LSTM model with improved architecture...")
                input_shape = (X_train.shape[1], X_train.shape[2])
                
                params = {
                    "lstm_units_1": 256,  # Increased capacity
                    "lstm_units_2": 128,  # Increased capacity
                    "lstm_units_3": 64,   # Additional layer
                    "dropout_rate": 0.2,  # Adjusted dropout
                    "learning_rate": 0.001,
                    "epochs": 100,        # Increased epochs
                    "batch_size": 64,     # Increased batch size
                    "patience": 15        # Increased patience
                }

                model = Sequential([
                    # First LSTM layer with batch normalization
                    Bidirectional(LSTM(params["lstm_units_1"], 
                                     return_sequences=True,
                                     kernel_initializer='he_normal')),
                    BatchNormalization(),
                    Dropout(params["dropout_rate"]),
                    
                    # Second LSTM layer
                    Bidirectional(LSTM(params["lstm_units_2"],
                                     return_sequences=True,
                                     kernel_initializer='he_normal')),
                    BatchNormalization(),
                    Dropout(params["dropout_rate"]),
                    
                    # Third LSTM layer
                    LSTM(params["lstm_units_3"],
                         kernel_initializer='he_normal'),
                    BatchNormalization(),
                    Dropout(params["dropout_rate"]),
                    
                    # Output layers with better regularization
                    Dense(32, activation='relu'),
                    BatchNormalization(),
                    Dense(1, activation='linear')
                ])

                optimizer = Adam(learning_rate=params["learning_rate"])
                model.compile(optimizer=optimizer, loss='huber')  # Huber loss for robustness

                # Enhanced callbacks
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=params["patience"],
                    restore_best_weights=True,
                    verbose=1
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.0001,
                    verbose=1
                )
                
                checkpoint = ModelCheckpoint(
                    filepath='/tmp/lstm_best_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )

                # Split validation set
                split = int(0.8 * len(X_train))
                X_tr, X_val = X_train[:split], X_train[split:]
                y_tr, y_val = y_train[:split], y_train[split:]

                # Training with callbacks
                model.fit(
                    X_tr, y_tr,
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, reduce_lr, checkpoint],
                    verbose=1
                )

                logger.info("LSTM training completed with improved architecture.")
                return model, params
                
            except Exception as e:
                logger.error(f"Error training LSTM model: {str(e)}")
                raise

        def predict_model(self, model, X_test, model_type):
            """Make predictions based on model type"""
            logger.info(f"Making predictions with {model_type} model...")
            if model_type == 'lstm':
                predictions = model.predict(X_test, verbose=0).flatten()
            else:
                predictions = model.predict(X_test)
            return predictions

        def save_predictions(self, results, model_type):
            """Save prediction results to GCS"""
            try:
                logger.info(f"Saving predictions for {model_type} model...")
                blob = self.bucket.blob(f'live_predictions/{model_type}/latest.json')
                blob.upload_from_string(json.dumps(results), content_type='application/json')
                logger.info(f"Predictions saved for {model_type} model.")
            except Exception as e:
                logger.error(f"Failed to save predictions for {model_type} model: {str(e)}")
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
                        else:
                            X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_data(data)

                        # Train model and make predictions
                        model, params = self.models[model_type](X_train, y_train)
                        predictions = self.predict_model(model, X_test, model_type)

                        # Calculate metrics
                        metrics = {
                            'mse': mean_squared_error(y_test, predictions),
                            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                            'r2': r2_score(y_test, predictions),
                            'mae': mean_absolute_error(y_test, predictions),
                            'mape': mean_absolute_percentage_error(y_test, predictions)
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

                        # Display metrics in Streamlit
                        st.subheader(f"📊 {model_type.capitalize()} Model Metrics")
                        st.json(metrics)

                        # Visualize Actual vs Predicted
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Actual'))
                        fig.add_trace(go.Scatter(x=test_dates, y=predictions, mode='lines', name='Predicted'))
                        fig.update_layout(title=f"Actual vs Predicted for {model_type.capitalize()} Model",
                                       xaxis_title='Date',
                                       yaxis_title='Price',
                                       legend=dict(x=0, y=1))
                        st.plotly_chart(fig)

                        logger.info(f"{model_type.capitalize()} Model Metrics: {metrics}")

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

    # Instantiate and run the StockPredictor
    try:
        predictor = StockPredictor(bucket)
        predictor.run_predictions()
    except Exception as e:
        st.error(f"Fatal error occurred: {str(e)}")
