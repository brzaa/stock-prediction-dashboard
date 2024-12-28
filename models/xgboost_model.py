import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from .model_utils import preprocess_data, save_model_results

class StockXGBoostModel:
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'reg:squarederror',
            'colsample_bytree': 0.3,
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10,
            'n_estimators': 100
        }
        self.model = None
        
    def train(self, X, y):
        """Train the XGBoost model"""
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model needs to be trained before making predictions")
        return self.model.predict(X)
        
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        return {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions)
        }
