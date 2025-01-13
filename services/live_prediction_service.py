import os
import pandas as pd
import numpy as np
from google.cloud import storage
import logging
import time
import psutil
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import warnings
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mlops_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DriftThresholds:
    alpha: float = 0.15
    beta: float = 0.10
    volume: float = 0.20
    price: float = 0.25
    window_size: int = 30
    min_data_points: int = 100

@dataclass
class ModelMetrics:
    mse: float
    rmse: float
    mae: float
    r2: float
    model_name: str
    timestamp: str = datetime.now().isoformat()

class EnhancedDataDriftManager:
    def __init__(self, bucket: storage.bucket.Bucket):
        self.thresholds = DriftThresholds()
        self.bucket = bucket
        self.reference_data = None
        self.untrained_data = pd.DataFrame()
        self.new_data_buffer = pd.DataFrame()
        self.drift_history: List[Dict] = []
        self.performance_metrics = {
            'detection_time': [],
            'processing_time': [],
            'memory_usage': []
        }

    def check_drift(self, new_data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        start_time = time.time()
        drift_detected = False
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'thresholds': asdict(self.thresholds),
            'severity': 'NONE'
        }

        try:
            if self.reference_data is None or len(new_data) < self.thresholds.min_data_points:
                return False, drift_report

            price_drift = self._check_price_drift(new_data)
            volume_drift = self._check_volume_drift(new_data)
            distribution_drift = self._check_distribution_drift(new_data)

            drift_report['metrics'] = {
                'price_drift': price_drift,
                'volume_drift': volume_drift,
                'distribution_drift': distribution_drift
            }

            if any(d > self.thresholds.alpha for d in [price_drift, volume_drift, distribution_drift]):
                drift_report['severity'] = 'HIGH'
                drift_detected = True
            elif any(d > self.thresholds.beta for d in [price_drift, volume_drift, distribution_drift]):
                drift_report['severity'] = 'MEDIUM'
                drift_detected = True

            self._log_performance_metrics(
                detection_time=time.time() - start_time,
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024
            )

            self._store_drift_history(drift_report)
            return drift_detected, drift_report

        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            raise

    def _check_price_drift(self, new_data: pd.DataFrame) -> float:
        price_cols = ['open', 'high', 'low', 'close']
        drift_magnitude = 0.0

        for col in price_cols:
            if col in self.reference_data.columns and col in new_data.columns:
                ref_stats = self._calculate_stats(self.reference_data[col])
                new_stats = self._calculate_stats(new_data[col])
                rel_change = abs(new_stats['mean'] - ref_stats['mean']) / ref_stats['std']
                drift_magnitude = max(drift_magnitude, rel_change)

        return drift_magnitude

    def _check_volume_drift(self, new_data: pd.DataFrame) -> float:
        if 'volume' not in new_data.columns:
            return 0.0

        ref_vol = self.reference_data['volume'].mean()
        new_vol = new_data['volume'].mean()
        return abs(new_vol - ref_vol) / ref_vol

    def _check_distribution_drift(self, new_data: pd.DataFrame) -> float:
        from scipy import stats
        max_ks_stat = 0.0

        for col in self.reference_data.columns:
            if col in new_data.columns:
                ks_stat, _ = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    new_data[col].dropna()
                )
                max_ks_stat = max(max_ks_stat, ks_stat)
        return max_ks_stat

    def _log_performance_metrics(self, detection_time: float, memory_usage: float):
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'detection_time': detection_time,
            'memory_usage': memory_usage
        }

        for key, value in metrics.items():
            if key != 'timestamp':
                self.performance_metrics[f'{key}'].append(value)

        self._store_performance_metrics(metrics)

    def _store_drift_history(self, drift_report: Dict[str, Any]):
        self.drift_history.append(drift_report)
        blob = self.bucket.blob(
            f'drift_history/{datetime.now().strftime("%Y/%m/%d/drift_%H%M%S.json")}'
        )
        blob.upload_from_string(json.dumps(drift_report))

    def _store_performance_metrics(self, metrics: Dict[str, Any]):
        blob = self.bucket.blob(
            f'performance_metrics/{datetime.now().strftime("%Y/%m/%d/perf_%H%M%S.json")}'
        )
        blob.upload_from_string(json.dumps(metrics))

    def _calculate_stats(self, series: pd.Series) -> Dict[str, float]:
        return {
            'mean': series.mean(),
            'std': series.std(),
        }

class EnhancedModelVersionControl:
    def __init__(self, bucket: storage.bucket.Bucket):
        self.bucket = bucket
        self.current_version = {
            'production': None,
            'staging': None,
            'development': None
        }
        self.version_history: List[Dict] = []
        self.promotion_history: List[Dict] = []

    def create_version(self, model: Any, metrics: ModelMetrics, metadata: Dict[str, Any]) -> str:
        version_id = f"v_{int(time.time())}_{metadata['model_type']}"
        version_info = {
            'id': version_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics),
            'metadata': metadata,
            'environment': 'development'
        }

        self._store_version_info(version_id, version_info)
        self._store_model_artifact(version_id, model)
        return version_id

    def promote_version(self, version_id: str, target_env: str, justification: str) -> bool:
        try:
            if target_env not in ['staging', 'production']:
                raise ValueError(f"Invalid target environment: {target_env}")

            version_info = self._load_version_info(version_id)
            if not version_info:
                raise ValueError(f"Version {version_id} not found")

            promotion_info = {
                'timestamp': datetime.now().isoformat(),
                'version_id': version_id,
                'from_env': version_info['environment'],
                'to_env': target_env,
                'justification': justification
            }

            version_info['environment'] = target_env
            self._store_version_info(version_id, version_info)
            self._store_promotion_record(promotion_info)
            self.current_version[target_env] = version_id
            return True

        except Exception as e:
            logger.error(f"Error promoting version {version_id}: {str(e)}")
            return False

    def _store_version_info(self, version_id: str, version_info: Dict[str, Any]):
        blob = self.bucket.blob(f'model_versions/{version_id}/info.json')
        blob.upload_from_string(json.dumps(version_info))

    def _store_model_artifact(self, version_id: str, model: Any):
        import pickle
        blob = self.bucket.blob(f'model_versions/{version_id}/model.pkl')
        blob.upload_from_string(pickle.dumps(model))

    def _store_promotion_record(self, promotion_info: Dict[str, Any]):
        self.promotion_history.append(promotion_info)
        blob = self.bucket.blob(
            f'promotions/{datetime.now().strftime("%Y/%m/%d/promotion_%H%M%S.json")}'
        )
        blob.upload_from_string(json.dumps(promotion_info))

    def _load_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        blob = self.bucket.blob(f'model_versions/{version_id}/info.json')
        if not blob.exists():
            return None
        return json.loads(blob.download_as_string())

class EnhancedStockPredictor:
    def __init__(self, bucket_name: str = "mlops-brza"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        self.drift_manager = EnhancedDataDriftManager(self.bucket)
        self.version_control = EnhancedModelVersionControl(self.bucket)
        
        self.scalers = {}
        self.current_models = {}
        self.model_metrics = {}

        self._initialize_storage()
        logger.info("Enhanced Stock Predictor initialized successfully")

    def _initialize_storage(self):
        required_folders = [
            'model_versions/',
            'drift_history/',
            'performance_metrics/',
            'promotions/',
            'predictions/',
            'training_metrics/',
            'evaluation_metrics/'
        ]
        for folder in required_folders:
            blob = self.bucket.blob(folder)
            if not blob.exists():
                blob.upload_from_string('')

    def fetch_stock_data(self) -> pd.DataFrame:
        try:
            blob = self.bucket.blob('stock_data/MASB_latest.csv')
            if not blob.exists():
                raise FileNotFoundError("File stock_data/MASB_latest.csv not found in the bucket.")

            local_file = '/tmp/MASB_latest.csv'
            blob.download_to_filename(local_file)
            
            data = pd.read_csv(local_file, parse_dates=['date'])
            data.set_index('date', inplace=True)

            logger.info(f"Fetched {len(data)} records from {blob.name}")
            return data

        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def _train_lightgbm(self, X_train_scaled: np.ndarray, y_train: pd.Series) -> LGBMRegressor:
        model = LGBMRegressor(
            objective='regression',
            max_depth=7,
            learning_rate=0.05,
            n_estimators=100,
            num_leaves=31
        )
        model.fit(X_train_scaled, y_train)
        return model

    def _train_xgboost(self, X_train_scaled: np.ndarray, y_train: pd.Series) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor(
            max_depth=7,
            learning_rate=0.05,
            n_estimators=100,
            objective='reg:squarederror'
        )
        model.fit(X_train_scaled, y_train)
        return model

    def _train_decision_tree(self, X_train_scaled: np.ndarray, y_train: pd.Series) -> DecisionTreeRegressor:
        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train)
        return model

    def _train_lstm(self, X_train_scaled: np.ndarray, y_train: pd.Series) -> Sequential:
        X_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5
        )
        
        model.fit(
            X_reshaped, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return model

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss
        
        try:
            models = {}
            training_metrics = {}
            X_train_scaled = {}
            
            # Initialize scalers for each model
            for model_name in ['lightgbm', 'xgboost', 'decision_tree', 'lstm']:
                self.scalers[model_name] = StandardScaler()
                X_train_scaled[model_name] = self.scalers[model_name].fit_transform(X_train)
            
            # Train each model
            models['lightgbm'] = self._train_lightgbm(X_train_scaled['lightgbm'], y_train)
            models['xgboost'] = self._train_xgboost(X_train_scaled['xgboost'], y_train)
            models['decision_tree'] = self._train_decision_tree(X_train_scaled['decision_tree'], y_train)
            models['lstm'] = self._train_lstm(X_train_scaled['lstm'], y_train)
            
            training_metrics = {
                'timestamp': datetime.now().isoformat(),
                'duration': time.time() - start_time,
                'memory_used': (psutil.Process().memory_info().rss - memory_before) / 1024 / 1024,
                'data_points': len(X_train),
                'feature_count': X_train.shape[1]
            }

            blob = self.bucket.blob(
                f'training_metrics/{datetime.now().strftime("%Y/%m/%d/training_%H%M%S.json")}'
            )
            blob.upload_from_string(json.dumps(training_metrics))

            return models

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def _evaluate_model(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, ModelMetrics]:
        try:
            metrics = {}
            
            for model_name, model in models.items():
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # For LSTM, reshape the input
                if model_name == 'lstm':
                    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                
                predictions = model.predict(X_test_scaled)
                
                metrics[model_name] = ModelMetrics(
                    mse=mean_squared_error(y_test, predictions),
                    rmse=np.sqrt(mean_squared_error(y_test, predictions)),
                    mae=mean_absolute_error(y_test, predictions),
                    r2=r2_score(y_test, predictions),
                    model_name=model_name
                )

                eval_log = {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_name,
                    'metrics': asdict(metrics[model_name]),
                    'test_size': len(X_test),
                    'prediction_stats': {
                        'mean': float(np.mean(predictions)),
                        'std': float(np.std(predictions)),
                        'min': float(np.min(predictions)),
                        'max': float(np.max(predictions))
                    }
                }

                blob = self.bucket.blob(
                    f'evaluation_metrics/{datetime.now().strftime("%Y/%m/%d/eval_%H%M%S_{model_name}.json")}'
                )
                blob.upload_from_string(json.dumps(eval_log))

            return metrics

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def train_and_evaluate(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, ModelMetrics]]:
        try:
            X_train, X_test, y_train, y_test = self._prepare_training_data(data)
            models = self._train_model(X_train, y_train)
            metrics = self._evaluate_model(models, X_test, y_test)

            # Store best model across all types
            best_model_name = max(metrics.items(), key=lambda x: x[1].r2)[0]
            self.current_models = models  # Store all models
            
            # Create version for each model
            for model_name, model in models.items():
                version_id = self.version_control.create_version(
                    model=model,
                    metrics=metrics[model_name],
                    metadata={
                        'model_type': model_name,
                        'data_points': len(data),
                        'training_date': datetime.now().isoformat(),
                        'feature_columns': list(X_train.columns),
                        'is_best_model': model_name == best_model_name
                    }
                )
                logger.info(f"Successfully trained and versioned {model_name} model: {version_id}")

            return models, metrics

        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise

    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Simple 80/20 split; adjust or enhance as needed."""
        split_idx = int(len(data) * 0.8)
        X = data.drop('close', axis=1)
        y = data['close']

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test

    def predict(self, data: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        try:
            drift_detected, drift_report = self.drift_manager.check_drift(data)
            predictions = {}
            
            for model_name, model in self.current_models.items():
                if model is None:
                    logger.warning(f"No current model loaded for {model_name}; returning zeros.")
                    predictions[model_name] = (np.zeros(len(data)), drift_report)
                    continue

                X = data.drop('close', axis=1, errors='ignore')
                X_scaled = self.scalers[model_name].transform(X)
                
                # Reshape for LSTM
                if model_name == 'lstm':
                    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                
                pred = model.predict(X_scaled)
                predictions[model_name] = (pred, drift_report)
                
                self._log_prediction(data, pred, drift_report, model_name)
            
            if drift_detected and drift_report['severity'] == 'HIGH':
                self._handle_high_severity_drift(data)
            
            return predictions

        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

    def _handle_high_severity_drift(self, data: pd.DataFrame):
        logger.info("Handling high severity drift - retraining all models")
        new_models, metrics = self.train_and_evaluate(data)
        
        # Find best performing model
        best_model_name = max(metrics.items(), key=lambda x: x[1].r2)[0]
        best_metrics = metrics[best_model_name]
        
        if best_metrics.r2 > 0.8:  # You can adjust this threshold
            version_id = self.version_control.create_version(
                model=new_models[best_model_name],
                metrics=best_metrics,
                metadata={
                    'model_type': best_model_name,
                    'drift_triggered': True,
                    'is_best_model': True
                }
            )
            
            self.version_control.promote_version(
                version_id=version_id,
                target_env='staging',
                justification=f'Drift-triggered retraining with good metrics (R² = {best_metrics.r2:.3f})'
            )

    def _log_prediction(self, data: pd.DataFrame, prediction: np.ndarray, drift_report: Dict[str, Any], model_name: str):
        prediction_log = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'prediction_summary': {
                'mean': float(np.mean(prediction)),
                'std': float(np.std(prediction)),
                'min': float(np.min(prediction)),
                'max': float(np.max(prediction))
            },
            'drift_detected': drift_report['severity'] != 'NONE',
            'model_version': self.version_control.current_version['production']
        }

        blob = self.bucket.blob(
            f'predictions/{datetime.now().strftime("%Y/%m/%d/pred_%H%M%S_{model_name}.json")}'
        )
        blob.upload_from_string(json.dumps(prediction_log))

    def get_system_status(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.now().isoformat(),
            'models_status': {
                model_name: {
                    'production': self.version_control.current_version['production'],
                    'staging': self.version_control.current_version['staging']
                }
                for model_name in ['lightgbm', 'xgboost', 'decision_tree', 'lstm']
            },
            'drift_status': {
                'total_checks': len(self.drift_manager.drift_history),
                'last_check': self.drift_manager.drift_history[-1] if self.drift_manager.drift_history else None
            },
            'data_status': {
                'reference_data_size': len(self.drift_manager.reference_data) if self.drift_manager.reference_data is not None else 0,
                'untrained_data_size': len(self.drift_manager.untrained_data)
            },
            'performance_metrics': {
                'avg_detection_time': np.mean(self.drift_manager.performance_metrics['detection_time']) if self.drift_manager.performance_metrics['detection_time'] else None,
                'avg_memory_usage': np.mean(self.drift_manager.performance_metrics['memory_usage']) if self.drift_manager.performance_metrics['memory_usage'] else None
            }
        }


def main():
    try:
        logger.info("Starting Enhanced Stock Prediction Pipeline")
        
        predictor = EnhancedStockPredictor(bucket_name="mlops-brza")

        # Initial system status
        initial_status = predictor.get_system_status()
        logger.info(f"Initial system status: {json.dumps(initial_status, indent=2)}")

        # Fetch data
        data = predictor.fetch_stock_data()
        logger.info(f"Fetched {len(data)} records")

        # Train initial models
        models, metrics = predictor.train_and_evaluate(data)
        logger.info("Initial model metrics:")
        for model_name, metric in metrics.items():
            logger.info(f"{model_name}: {asdict(metric)}")

        # Set up continuous monitoring
        while True:
            try:
                new_data = predictor.fetch_stock_data()
                predictions = predictor.predict(new_data)
                
                current_status = predictor.get_system_status()
                logger.info(f"Current system status: {json.dumps(current_status, indent=2)}")
                
                # Log predictions for each model
                for model_name, (pred, _) in predictions.items():
                    logger.info(f"{model_name} predictions summary: mean={np.mean(pred):.2f}, std={np.std(pred):.2f}")
                
                time.sleep(300)  # 5 minutes interval

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # wait 1 minute before retrying
                continue

    except Exception as e:
        logger.error(f"Fatal error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
