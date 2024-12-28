
import pandas as pd
from google.cloud import storage
import json
from datetime import datetime

class DataLoader:
    def __init__(self, bucket_name="mlops-brza"):
        """Initialize with GCS bucket connection"""
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def load_model_results(self, run_id):
        """Load specific model run results"""
        try:
            blob = self.bucket.blob(f'model_outputs/{run_id}/results.json')
            return json.loads(blob.download_as_string())
        except Exception as e:
            raise Exception(f"Error loading results: {str(e)}")

    def load_historical_runs(self):
        """Load all historical run results"""
        runs = []
        for blob in self.bucket.list_blobs(prefix='model_outputs/'):
            if blob.name.endswith('results.json'):
                try:
                    data = json.loads(blob.download_as_string())
                    runs.append(data)
                except Exception:
                    continue
        return runs

    def load_latest_data(self):
        """Load the most recent stock data"""
        try:
            blob = self.bucket.blob('stock_data/MASB.csv')
            return pd.read_csv(blob.download_as_string())
        except Exception as e:
            raise Exception(f"Error loading latest data: {str(e)}")
