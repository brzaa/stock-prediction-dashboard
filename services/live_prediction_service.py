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
        """
        Enhanced drift detection with multiple thresholds and performance tracking
        """
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

            # Check different types of drift
            price_drift = self._check_price_drift(new_data)
            volume_drift = self._check_volume_drift(new_data)
            distribution_drift = self._check_distribution_drift(new_data)

            drift_report['metrics'] = {
                'price_drift': price_drift,
                'volume_drift': volume_drift,
                'distribution_drift': distribution_drift
            }

            # Determine severity
            if any(d > self.thresholds.alpha for d in [price_drift, volume_drift, distribution_drift]):
                drift_report['severity'] = 'HIGH'
                drift_detected = True
            elif any(d > self.thresholds.beta for d in [price_drift, volume_drift, distribution_drift]):
                drift_report['severity'] = 'MEDIUM'
                drift_detected = True

            # Log performance metrics
            self._log_performance_metrics(
                detection_time=time.time() - start_time,
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024
            )

            # Store drift history
            self._store_drift_history(drift_report)

            return drift_detected, drift_report

        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            raise

    def _check_price_drift(self, new_data: pd.DataFrame) -> float:
        """Check for drift in price-related features"""
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
        """Check for drift in trading volume"""
        if 'volume' not in new_data.columns:
            return 0.0

        ref_vol = self.reference_data['volume'].mean()
        new_vol = new_data['volume'].mean()
        
        return abs(new_vol - ref_vol) / ref_vol

    def _check_distribution_drift(self, new_data: pd.DataFrame) -> float:
        """Check for changes in overall distribution using KS test"""
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
        """Log performance metrics of drift detection"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'detection_time': detection_time,
            'memory_usage': memory_usage
        }

        # Store locally
        for key, value in metrics.items():
            if key != 'timestamp':
                self.performance_metrics[f'{key}'].append(value)

        # Store in GCS
        self._store_performance_metrics(metrics)

    def _store_drift_history(self, drift_report: Dict[str, Any]):
        """Store drift detection history in GCS"""
        self.drift_history.append(drift_report)
        
        # Store in GCS
        blob = self.bucket.blob(
            f'drift_history/{datetime.now().strftime("%Y/%m/%d/drift_%H%M%S.json")}'
        )
        blob.upload_from_string(json.dumps(drift_report))

    def _store_performance_metrics(self, metrics: Dict[str, Any]):
        """Store performance metrics in GCS"""
        blob = self.bucket.blob(
            f'performance_metrics/{datetime.now().strftime("%Y/%m/%d/perf_%H%M%S.json")}'
        )
        blob.upload_from_string(json.dumps(metrics))

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
        """Create new model version with enhanced tracking"""
        version_id = f"v_{int(time.time())}"
        version_info = {
            'id': version_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics),
            'metadata': metadata,
            'environment': 'development'
        }

        # Store version info
        self._store_version_info(version_id, version_info)
        
        # Store model artifact
        self._store_model_artifact(version_id, model)
        
        return version_id

    def promote_version(self, version_id: str, target_env: str, justification: str) -> bool:
        """Promote model version with validation and logging"""
        try:
            if target_env not in ['staging', 'production']:
                raise ValueError(f"Invalid target environment: {target_env}")

            version_info = self._load_version_info(version_id)
            if not version_info:
                raise ValueError(f"Version {version_id} not found")

            # Log promotion
            promotion_info = {
                'timestamp': datetime.now().isoformat(),
                'version_id': version_id,
                'from_env': version_info['environment'],
                'to_env': target_env,
                'justification': justification
            }

            # Update version info
            version_info['environment'] = target_env
            self._store_version_info(version_id, version_info)
            
            # Store promotion record
            self._store_promotion_record(promotion_info)
            
            # Update current version pointer
            self.current_version[target_env] = version_id
            
            return True

        except Exception as e:
            logger.error(f"Error promoting version {version_id}: {str(e)}")
            return False

    def _store_version_info(self, version_id: str, version_info: Dict[str, Any]):
        """Store version information in GCS"""
        blob = self.bucket.blob(f'model_versions/{version_id}/info.json')
        blob.upload_from_string(json.dumps(version_info))

    def _store_model_artifact(self, version_id: str, model: Any):
        """Store model artifact in GCS"""
        import pickle
        blob = self.bucket.blob(f'model_versions/{version_id}/model.pkl')
        blob.upload_from_string(pickle.dumps(model))

    def _store_promotion_record(self, promotion_info: Dict[str, Any]):
        """Store promotion record in GCS"""
        self.promotion_history.append(promotion_info)
        blob = self.bucket.blob(
            f'promotions/{datetime.now().strftime("%Y/%m/%d/promotion_%H%M%S.json")}'
        )
        blob.upload_from_string(json.dumps(promotion_info))

    def _load_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Load version information from GCS"""
        blob = self.bucket.blob(f'model_versions/{version_id}/info.json')
        if not blob.exists():
            return None
        return json.loads(blob.download_as_string())

class EnhancedStockPredictor:
    def __init__(self, bucket_name: str = "mlops-stock-predictions"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Initialize enhanced components
        self.drift_manager = EnhancedDataDriftManager(self.bucket)
        self.version_control = EnhancedModelVersionControl(self.bucket)
        
        # Initialize model-specific components
        self.scaler = StandardScaler()
        self.current_model = None
        self.model_metrics = None

        self._initialize_storage()
        logger.info("Enhanced Stock Predictor initialized successfully")

    def _initialize_storage(self):
        """Initialize GCS storage structure"""
        required_folders = [
            'model_versions/',
            'drift_history/',
            'performance_metrics/',
            'promotions/',
            'predictions/'
        ]

        for folder in required_folders:
            blob = self.bucket.blob(folder)
            if not blob.exists():
                blob.upload_from_string('')

    def train_and_evaluate(self, data: pd.DataFrame) -> Tuple[Any, ModelMetrics]:
        """Train model with enhanced monitoring and validation"""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self._prepare_training_data(data)
            
            # Train model
            model = self._train_model(X_train, y_train)
            
            # Evaluate model
            metrics = self._evaluate_model(model, X_test, y_test)
            
            # Create new version
            version_id = self.version_control.create_version(
                model=model,
                metrics=metrics,
                metadata={
                    'data_points': len(data),
                    'training_date': datetime.now().isoformat(),
                    'feature_columns': list(X_train.columns)
                }
            )

            logger.info(f"Successfully trained and versioned model: {version_id}")
            return model, metrics

        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions with enhanced monitoring"""
        try:
            # Check for drift
            drift_detected, drift_report = self.drift_manager.check_drift(data)
            
            if drift_detected:
                logger.warning("Drift detected during prediction")
                if drift_report['severity'] == 'HIGH':
                    # Trigger retraining
                    self._handle_high_severity_drift(data)
            
            # Make prediction
            prediction = self._make_prediction(data)
            
            # Log prediction
            self._log_prediction(data, prediction, drift_report)
            
            return prediction, drift_report

        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

    def _handle_high_severity_drift(self, data: pd.DataFrame):
        """Handle high severity drift detection"""
        logger.info("Handling high severity drift")
        
        # Train new model
        new_model, metrics = self.train_and_evaluate(data)
        
        # Create new version
        version_id = self.version_control.create_version(
            model=new_model,
            metrics=metrics,
            metadata={'drift_triggered': True}
        )
        
        # Promote to staging if metrics are good
        if metrics.r2 > 0.8:  # Example threshold
            self.version_control.promote_version(
                version_id=version_id,
                target_env='staging',
                justification='Drift-triggered retraining with good metrics'
            )

    def _log_prediction(self, data: pd.DataFrame, prediction: np.ndarray, drift_report: Dict[str, Any]):
        """Log prediction details"""
        prediction_log = {
            'timestamp': datetime.now().isoformat(),
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
            f'predictions/{datetime.now().strftime("%Y/%m/%d/pred_%H%M%S.json")}'
        )
        blob.upload_from_string(json.dumps(prediction_log))

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train model with enhanced monitoring"""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss

        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            model = LGBMRegressor(
                objective='regression',
                max_depth=7,
                learning_rate=0.05,
                n_estimators=100,
                num_leaves=31
            )
            model.fit(X_train_scaled, y_train)

            # Log training metrics
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

            return model

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate model with enhanced metrics"""
        try:
            X_test_scaled = self.scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)

            metrics = ModelMetrics(
                mse=mean_squared_error(y_test, predictions),
                rmse=np.sqrt(mean_squared_error(y_test, predictions)),
                mae=mean_absolute_error(y_test, predictions),
                r2=r2_score(y_test, predictions)
            )

            # Log evaluation metrics
            eval_log = {
                'timestamp': datetime.now().isoformat(),
                'metrics': asdict(metrics),
                'test_size': len(X_test),
                'prediction_stats': {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions))
                }
            }

            blob = self.bucket.blob(
                f'evaluation_metrics/{datetime.now().strftime("%Y/%m/%d/eval_%H%M%S.json")}'
            )
            blob.upload_from_string(json.dumps(eval_log))

            return metrics

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_model': {
                'production': self.version_control.current_version['production'],
                'staging': self.version_control.current_version['staging']
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
    """Main execution function with enhanced error handling and logging"""
    try:
        logger.info("Starting Enhanced Stock Prediction Pipeline")
        predictor = EnhancedStockPredictor()

        # Initial system status
        initial_status = predictor.get_system_status()
        logger.info(f"Initial system status: {json.dumps(initial_status, indent=2)}")

        # Fetch and process data
        data = predictor.fetch_stock_data()
        logger.info(f"Fetched {len(data)} records")

        # Train initial model
        model, metrics = predictor.train_and_evaluate(data)
        logger.info(f"Initial model metrics: {asdict(metrics)}")

        # Set up continuous monitoring
        while True:
            try:
                # Fetch new data
                new_data = predictor.fetch_stock_data()
                
                # Make predictions
                predictions, drift_report = predictor.predict(new_data)
                
                # Get updated system status
                current_status = predictor.get_system_status()
                logger.info(f"Current system status: {json.dumps(current_status, indent=2)}")
                
                # Wait for next iteration
                time.sleep(300)  # 5 minutes interval

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
                continue

    except Exception as e:
        logger.error(f"Fatal error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
