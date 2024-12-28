import mlflow
import mlflow.xgboost
from google.cloud import storage
from models.xgboost_model import StockXGBoostModel
from models.model_utils import preprocess_data, save_model_results
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_path, project_id, bucket_name, split_date):
    """Main training function"""
    try:
        # Load data
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(data_path)
        data = pd.read_csv(blob.download_as_string(), parse_dates=['date'], index_col='date')
        
        # Preprocess data
        (X_train, y_train), (X_test, y_test) = preprocess_data(data, split_date)
        
        # Train model
        with mlflow.start_run() as run:
            model = StockXGBoostModel()
            model.train(X_train, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save results
            save_model_results(model, X_test, y_test, metrics, run.info.run_id, bucket_name)
            
            return run.info.run_id, model, metrics
            
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise
