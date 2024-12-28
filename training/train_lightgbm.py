import os
import mlflow
import mlflow.lightgbm
import pandas as pd
import talib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from google.cloud import storage
import logging
from typing import Tuple, Any
from datetime import datetime
import pickle
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_model_results(run_id: str, results: dict, bucket_name: str = "mlops-brza"):
    """Save model results to GCS"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Save results.json in model_outputs/lightgbm/<run_id>/
        output_path = f"model_outputs/lightgbm/{run_id}/results.json"
        blob = bucket.blob(output_path)
        blob.upload_from_string(
            json.dumps(results, indent=2),
            content_type='application/json'
        )
        
        logger.info(f"Results saved to: gs://{bucket_name}/{output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

def train_and_log_model_colab(
    project_id: str,
    bucket_name: str = "mlops-brza",
    split_date: str = '2022-01-01'
) -> Tuple[str, Any]:
    with mlflow.start_run() as run:
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info("Fetching data from GCS...")
        
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob('stock_data/MASB.csv')
        data = pd.read_csv(blob.open(mode='rb'), parse_dates=['date'], index_col='date')

        # Data preprocessing
        data.columns = data.columns.str.lower()
        data['close'] = pd.to_numeric(data['close'], errors='coerce')

        # Feature engineering
        data['log_return'] = data['close'].pct_change().apply(lambda x: np.log(1 + x))
        data['moving_avg_10'] = data['close'].rolling(window=10).mean()
        data['std_dev_10'] = data['close'].rolling(window=10).std()
        data['rsi'] = talib.RSI(data['close'].values, timeperiod=14)
        data['macd'], data['signal'], data['hist'] = talib.MACD(data['close'].values)

        for lag in range(1, 6):
            data[f'lag_{lag}'] = data['close'].shift(lag)

        data = data.ffill().bfill().dropna()

        split_date = pd.Timestamp(split_date)
        train_data = data.loc[data.index < split_date]
        test_data = data.loc[data.index >= split_date]

        X_train = train_data.drop(columns=['close'])
        y_train = train_data['close']
        X_test = test_data.drop(columns=['close'])
        y_test = test_data['close']

        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_params = {
            'num_leaves': 31,
<<<<<<< HEAD
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_samples': 5,
=======
            'max_depth': 10,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_samples': 20,
>>>>>>> 63bba2408c5524e16776d20ba93512706bf4185a
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 1e-5,
            'random_state': 42
        }

        logger.info("Training LightGBM model...")
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })

        results = {
            'run_id': run.info.run_id,
            'model_type': 'lightgbm',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'lightgbm',  # Add model type
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mae': float(mae)
            },
            'parameters': best_params,
            'predictions': y_pred.tolist(),
            'actual_values': y_test.tolist(),
<<<<<<< HEAD
            'feature_names': list(X_test.columns),
            'feature_importance': model.feature_importances_.tolist(),
            'dates': test_data.index.strftime('%Y-%m-%d').tolist()
=======
            'feature_names': list(X_test.columns)
>>>>>>> 63bba2408c5524e16776d20ba93512706bf4185a
        }
        
        save_model_results(run.info.run_id, results, bucket_name)

        # Save model locally first
        model_path = "/content/models"
        os.makedirs(model_path, exist_ok=True)
<<<<<<< HEAD

        # Save the LightGBM model
        model_save_path = f"{model_path}/lightgbm_model.txt"
        model.booster_.save_model(model_save_path)
        
        # Upload to GCS
        model_blob = bucket.blob(f'models/lightgbm/{run.info.run_id}/model.txt')
        model_blob.upload_from_filename(model_save_path)

        # Log model with MLflow
        mlflow.lightgbm.log_model(
            model,
            "model",
            registered_model_name="stock_prediction_lgb_model"
        )

        # Print run ID clearly
        print("\n" + "="*50)
        print(f"Training completed successfully!")
        print(f"Run ID for dashboard: {run.info.run_id}")
        print("="*50 + "\n")
        
        logger.info(f"Run ID: {run.info.run_id}")
=======

        # Save the model using pickle
        model_save_path = f"{model_path}/lightgbm_model.pkl"
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)

        model_blob = bucket.blob(f'models/lightgbm/{run.info.run_id}/lightgbm_model.pkl')
        model_blob.upload_from_filename(model_save_path)

>>>>>>> 63bba2408c5524e16776d20ba93512706bf4185a
        logger.info(f"Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")

        return run.info.run_id, scaler, model, X_test_scaled, y_test

if __name__ == "__main__":
    project_id = "mlops-thesis"
    try:
        run_id, _, _, _, _ = train_and_log_model_colab(project_id)
        print(f"\nUse this Run ID in the dashboard: {run_id}\n")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
